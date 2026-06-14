"""
Testes de regressão do núcleo determinístico FLARE26.

Cobrem a parte que o produto vende como "confiável e determinística":
normalização numérica BR, sanitização de OCR e o juiz simbólico M4.
Nenhum teste aqui toca rede ou IA — tudo é puro e reproduzível.
"""

import json
from pathlib import Path

import pytest

import flare26_core as core


# --------------------------------------------------------------------------
# Normalização numérica BR
# --------------------------------------------------------------------------
class TestExtrairNumerosBR:
    @pytest.mark.parametrize("texto, esperado", [
        ("R$ 1.200.000,50", [1200000.5]),
        ("multa de 12,5%", [12.5]),
        ("20%", [20.0]),
        ("0,5", [0.5]),
        ("5000.00", [5000.0]),
        ("1,2 bilhões", [1_200_000_000.0]),
        ("500 mil", [500_000.0]),
        ("3 milhões de reais", [3_000_000.0]),
        ("vinte por cento", []),          # texto por extenso não é capturado
        ("sem números aqui", []),
        ("", []),
    ])
    def test_casos(self, texto, esperado):
        assert core.extrair_numeros_br(texto) == esperado

    def test_dedup_e_ordenacao(self):
        assert core.extrair_numeros_br("10, 10 e 5") == [5.0, 10.0]

    def test_por_cento_equivale_a_percent(self):
        assert core.extrair_numeros_br("15 por cento") == core.extrair_numeros_br("15%")

    def test_entrada_invalida_nao_quebra(self):
        assert core.extrair_numeros_br(None) == []
        assert core.extrair_numeros_br(12345) == []


# --------------------------------------------------------------------------
# Sanitizador de OCR
# --------------------------------------------------------------------------
class TestSanitizar:
    def test_remove_nulos(self):
        assert "\x00" not in core.sanitizar_texto_pdf("a\x00b")

    def test_junta_hifenizacao(self):
        assert core.sanitizar_texto_pdf("contra-\nto") == "contrato"

    def test_colapsa_espacos(self):
        assert core.sanitizar_texto_pdf("a    b\t\tc") == "a b c"

    def test_vazio(self):
        assert core.sanitizar_texto_pdf("") == ""
        assert core.sanitizar_texto_pdf(None) == ""

    def test_idempotente(self):
        t = "Cláusula  multa\n\n\n\nde 10%"
        once = core.sanitizar_texto_pdf(t)
        assert core.sanitizar_texto_pdf(once) == once


# --------------------------------------------------------------------------
# Juiz simbólico M4
# --------------------------------------------------------------------------
class TestComparaSimbolico:
    def test_ambos_ausentes_lacuna(self):
        v, _ = core.comparar_simbolico(core.NAO_LOCALIZADO, "", core.NAO_LOCALIZADO, "")
        assert v == core.LACUNA_EVIDENCIA

    def test_um_ausente_divergencia(self):
        v, _ = core.comparar_simbolico("10%", "atraso", core.NAO_LOCALIZADO, "")
        assert v == core.DIVERGENCIA
        v, _ = core.comparar_simbolico(core.NAO_LOCALIZADO, "", "10%", "atraso")
        assert v == core.DIVERGENCIA

    def test_strings_identicas_consenso(self):
        v, _ = core.comparar_simbolico("10%", "por atraso", "10%", "POR ATRASO")
        assert v == core.CONSENSO

    def test_divergencia_matematica(self):
        v, motivos = core.comparar_simbolico("12,5%", "x", "20%", "y")
        assert v == core.DIVERGENCIA
        assert "divergência matemática" in motivos[0].lower()

    def test_numeros_iguais_condicao_diferente_vai_para_neural(self):
        # Mesmo número, condições textuais diferentes: simbólico não conclui.
        v, _ = core.comparar_simbolico("10%", "atraso", "10%", "inexecução")
        assert v == core.INDETERMINADO

    def test_economiza_api_em_divergencia_obvia(self):
        # Garante que o caminho barato (sem IA) é tomado.
        v, _ = core.comparar_simbolico("R$ 5.000,00", "a", "R$ 9.000,00", "b")
        assert v == core.DIVERGENCIA


# --------------------------------------------------------------------------
# Pontuação léxica configurável (generalização M1.5)
# --------------------------------------------------------------------------
class TestPontuacaoLexica:
    def test_default_pontua_termo_critico(self):
        s = core.pontuar_relevancia_lexica("cláusula de multa por atraso", [])
        assert s >= core.GatilhosLexicais().peso_critico

    def test_dominio_customizado(self):
        # Domínio agnóstico: trocamos os gatilhos sem mexer no código.
        g = core.GatilhosLexicais(criticos=("dosagem", "contraindicação"),
                                   comuns=(), hiper=())
        s = core.pontuar_relevancia_lexica("dosagem máxima diária", [], g)
        assert s == g.peso_critico


# --------------------------------------------------------------------------
# REGRESSÃO: re-julga as 98 extrações reais do ledger histórico
# --------------------------------------------------------------------------
FIXTURE = Path(__file__).parent / "fixtures" / "ledger_snapshot.json"


def _carregar_casos_ledger():
    if not FIXTURE.exists():
        return []
    registros = json.loads(FIXTURE.read_text(encoding="utf-8"))
    casos = []
    for r in registros:
        cv = r.get("caixa_de_vidro", {})
        ext = cv.get("M2_Extrator", {})
        a, b = ext.get("doc_A"), ext.get("doc_B")
        juiz = cv.get("M4_Juiz", {})
        if a and b and juiz.get("diagnostico"):
            casos.append((r.get("id_teste", "?"), a, b, juiz["diagnostico"]))
    return casos


CASOS_LEDGER = _carregar_casos_ledger()


@pytest.mark.skipif(not CASOS_LEDGER, reason="snapshot do ledger ausente")
def test_ledger_snapshot_tem_massa():
    # Sanidade: o dataset de validação não pode ter encolhido silenciosamente.
    # Apenas registros no schema novo (M2_Extrator doc_A/doc_B) são comparáveis;
    # o snapshot histórico mistura schemas antigos.
    assert len(CASOS_LEDGER) >= 5


@pytest.mark.skipif(not CASOS_LEDGER, reason="snapshot do ledger ausente")
@pytest.mark.parametrize("caso_id, a, b, diag_historico", CASOS_LEDGER,
                         ids=[c[0] for c in CASOS_LEDGER])
def test_juiz_simbolico_nao_contradiz_divergencia_numerica(caso_id, a, b, diag_historico):
    """O juiz simbólico atual não pode CONTRADIZER uma divergência numérica
    flagrante que já constava no histórico.

    Schema antigo do ledger usa 'valor_formatado'/'condicao_ou_prazo'.
    """
    resp_a = a.get("valor_formatado", core.NAO_LOCALIZADO)
    resp_b = b.get("valor_formatado", core.NAO_LOCALIZADO)
    cond_a = a.get("condicao_ou_prazo", "")
    cond_b = b.get("condicao_ou_prazo", "")

    veredito, _ = core.comparar_simbolico(resp_a, cond_a, resp_b, cond_b)

    nums_a = core.extrair_numeros_br(resp_a)
    nums_b = core.extrair_numeros_br(resp_b)
    ha_divergencia_numerica = bool(nums_a and nums_b and not set(nums_a) & set(nums_b))

    # Se há divergência numérica real, o juiz JAMAIS pode dizer CONSENSO.
    if ha_divergencia_numerica:
        assert veredito != core.CONSENSO, (
            f"{caso_id}: simbólico deu CONSENSO apesar de {nums_a} vs {nums_b}"
        )
