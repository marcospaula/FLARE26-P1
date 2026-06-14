"""
flare26_core — Núcleo determinístico e testável do FLARE26-P1.

Este módulo concentra a lógica simbólica (sem IA, sem rede, sem Streamlit)
que sustenta a "Caixa de Vidro": sanitização de texto, normalização numérica
em português BR e o juiz determinístico M4.

Princípios de projeto:
  * Importável sem efeitos colaterais (não lê chaves de API, não abre Chroma).
  * Determinístico: mesma entrada → mesma saída, sempre.
  * Agnóstico de domínio: heurísticas léxicas vivem em `GatilhosLexicais`,
    injetáveis em vez de fixas no código (generalização do M1.5/M4).

Ambos os apps Streamlit podem importar daqui sem reescrever a lógica.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

NAO_LOCALIZADO = "NÃO LOCALIZADO"
LACUNA = "LACUNA DE EVIDÊNCIA"

# Vereditos canônicos do juiz M4
CONSENSO = "CONSENSO TOTAL"
DIVERGENCIA = "DIVERGÊNCIA CRÍTICA"
LACUNA_EVIDENCIA = "LACUNA DE EVIDÊNCIA"
# Sinaliza que a lógica simbólica não foi suficiente e cabe fallback neural.
INDETERMINADO = "INDETERMINADO"


# ==========================================================================
# SANITIZADOR DE OCR (BLINDAGEM LÉXICA)
# ==========================================================================
def sanitizar_texto_pdf(texto_bruto: str) -> str:
    """Remove ruídos de OCR que destroem a matemática vetorial.

    Corrige caracteres nulos, quebras de linha excessivas, hifenização de
    fim de linha e espaços múltiplos. É idempotente.
    """
    if not texto_bruto:
        return ""

    texto = texto_bruto.replace("\x00", "")
    # Junta palavras hifenizadas quebradas no fim da linha ("contra-\nto")
    texto = re.sub(r"-\n+", "", texto)
    # Colapsa 3+ quebras de linha em parágrafo único (preserva parágrafos)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    # Remove tabs e espaços múltiplos dentro de uma frase
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto.strip()


# ==========================================================================
# NORMALIZAÇÃO NUMÉRICA PORTUGUÊS BR (A matemática não mente)
# ==========================================================================
_ESCALAS = [
    (r"bilh[õo]es?|bilh[aã]o|\bbil\b|\bbi\b", 1_000_000_000),
    (r"milh[õo]es?|milh[aã]o|\bmi\b", 1_000_000),
    (r"\bmil\b", 1_000),
]

# Aceita "1.200.000,50" (BR com separador de milhar), "5000.00", "5000" e
# "0,5"/"0.5". O primeiro ramo EXIGE ao menos um grupo ".ddd" (por isso `+`),
# senão ele canibalizaria "5000" como "500" e fragmentaria o número.
_NUM_PAT = r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"


def _parse_br(s: str) -> float:
    """Converte um literal numérico BR/US para float.

    "1.200.000,50" -> 1200000.5 ; "0,5" -> 0.5 ; "5000.00" -> 5000.0
    """
    s = s.strip()
    # Formato BR clássico com separador de milhar: 1.234.567(,89)?
    if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", s):
        return float(s.replace(".", "").replace(",", "."))
    # Caso geral: vírgula é decimal BR
    return float(s.replace(",", "."))


def extrair_numeros_br(texto: str) -> list[float]:
    """Extrai e normaliza todos os valores numéricos de um texto PT-BR.

    Suporta formato BR (R$ 1.200.000,50 → 1200000.5), escalas linguísticas
    (1,2 bilhões → 1.2e9; 500 mil → 500000.0) e "por cento" ≡ "%".
    Retorna lista ordenada e deduplicada de floats.
    """
    if not isinstance(texto, str):
        return []

    t = texto.lower().replace("por cento", "%").replace("r$", "").replace("reais", "")

    resultados: set[float] = set()
    for m in re.finditer(rf"({_NUM_PAT})", t):
        try:
            val = _parse_br(m.group(1))
        except ValueError:
            continue

        multiplicador = 1
        trecho_depois = t[m.end():m.end() + 20]
        for pattern, mult in _ESCALAS:
            if re.match(rf"\s*(?:{pattern})", trecho_depois):
                multiplicador = mult
                break

        resultados.add(round(val * multiplicador, 2))

    return sorted(resultados)


# ==========================================================================
# CONFIGURAÇÃO DE DOMÍNIO (generalização do M1.5)
# ==========================================================================
@dataclass(frozen=True)
class GatilhosLexicais:
    """Heurísticas léxicas do filtro M1.5, agora injetáveis por domínio.

    Os defaults cobrem o domínio de contratos/editais BR. Para outro domínio
    (saúde, engenharia, etc.) basta instanciar com outras listas — sem tocar
    no código do filtro.
    """
    comuns: tuple[str, ...] = (
        "contrato", "objeto", "pagamento", "valor", "percentual", "%",
    )
    criticos: tuple[str, ...] = (
        "multa", "penalidade", "sanção", "inexecução", "atraso", "infração",
        "descumprimento", "violação", "mora", "compensatória",
    )
    hiper: tuple[str, ...] = (
        "inexecução", "multa", "penalidade", "compensatória",
    )
    peso_comum: int = 1
    peso_critico: int = 3
    peso_hiper: int = 5
    corte_minimo: int = 3


def pontuar_relevancia_lexica(conteudo: str, termos_pergunta: list[str],
                              gatilhos: GatilhosLexicais | None = None) -> int:
    """Pontua um trecho pela relevância léxica (fallback BM25-like do M1.5).

    Determinístico e agnóstico: os termos de domínio vêm de `gatilhos`.
    """
    g = gatilhos or GatilhosLexicais()
    conteudo_lower = conteudo.lower()
    score = 0
    for termo in termos_pergunta:
        if termo in conteudo_lower:
            score += 1
    for comum in g.comuns:
        if comum in conteudo_lower:
            score += g.peso_comum
    for critico in g.criticos:
        if critico in conteudo_lower:
            score += g.peso_critico
    for hiper in g.hiper:
        if hiper in conteudo_lower:
            score += g.peso_hiper
    return score


# ==========================================================================
# JUIZ DETERMINÍSTICO M4 (núcleo simbólico, sem IA)
# ==========================================================================
def comparar_simbolico(resp_a: str, cond_a: str,
                       resp_b: str, cond_b: str) -> tuple[str, list[str]]:
    """Juiz simbólico do M4: decide o veredito sem chamar nenhuma IA.

    Retorna (veredito, motivos). O veredito INDETERMINADO sinaliza que a
    lógica simbólica não foi conclusiva e cabe ao chamador acionar o
    fallback neural (LLM). Isso é o "cost-killer": só gasta API quando a
    matemática/strings não bastam.
    """
    a, b = (resp_a or "").strip(), (resp_b or "").strip()

    # 1. Existência
    if a == NAO_LOCALIZADO and b == NAO_LOCALIZADO:
        return LACUNA_EVIDENCIA, ["Simbólico: nenhum documento possui a resposta."]
    if a == NAO_LOCALIZADO:
        return DIVERGENCIA, ["Simbólico: apenas o Documento B contém a informação."]
    if b == NAO_LOCALIZADO:
        return DIVERGENCIA, ["Simbólico: apenas o Documento A contém a informação."]

    # 2. Igualdade exata de string (resposta + condicionante)
    if a.lower() == b.lower() and (cond_a or "").strip().lower() == (cond_b or "").strip().lower():
        return CONSENSO, ["Simbólico: respostas e condicionantes são idênticas."]

    # 3. Divergência matemática flagrante
    nums_a = extrair_numeros_br(a)
    nums_b = extrair_numeros_br(b)
    if nums_a and nums_b and not set(nums_a).intersection(nums_b):
        return DIVERGENCIA, [
            f"Simbólico: divergência matemática detectada ({nums_a} vs {nums_b}). "
            "Nenhuma chamada de API foi necessária."
        ]

    # 4. Inconclusivo → caller deve acionar o juiz neural
    return INDETERMINADO, ["Simbólico inconclusivo: requer julgamento neural (M4 LLM)."]


# ==========================================================================
# JUIZ N-WAY (consenso por agrupamento) — escala A-vs-B para N documentos
# ==========================================================================
@dataclass(frozen=True)
class GrupoConsenso:
    """Um agrupamento de documentos que extraíram a mesma resposta."""
    valor: str                 # rótulo legível do valor representativo do grupo
    documentos: tuple[str, ...]
    numeros: tuple[float, ...]  # números normalizados (vazio se resposta textual)


@dataclass(frozen=True)
class ResultadoConsenso:
    """Veredito agregado do M4 sobre N documentos."""
    veredito: str
    grupos: tuple[GrupoConsenso, ...]   # grupos com dado, do maior para o menor
    lacunas: tuple[str, ...]            # documentos sem a informação
    resumo: str


def _chave_canonica(resposta: str) -> tuple:
    """Chave determinística de equivalência de uma resposta.

    - "NÃO LOCALIZADO" → sentinela de lacuna
    - resposta com números → equivalência pelo CONJUNTO de números (12,5% ≡ 12.5)
    - resposta textual → string normalizada (minúscula, espaços colapsados)
    """
    r = (resposta or "").strip()
    if not r or r.upper() == NAO_LOCALIZADO:
        return ("__LACUNA__",)
    nums = extrair_numeros_br(r)
    if nums:
        return ("num", frozenset(nums))
    return ("txt", re.sub(r"\s+", " ", r.lower()))


def comparar_n_documentos(extracoes: dict[str, str]) -> ResultadoConsenso:
    """Juiz N-way determinístico: agrupa documentos por resposta equivalente.

    `extracoes` mapeia nome_do_documento → resposta_direta extraída (M2).
    Retorna grupos de consenso, lacunas e um veredito agregado. Tudo O(N),
    sem nenhuma chamada de IA (cost-killer também na escala).
    """
    grupos_map: dict[tuple, list[str]] = {}
    lacunas: list[str] = []

    for nome, resposta in extracoes.items():
        chave = _chave_canonica(resposta)
        if chave == ("__LACUNA__",):
            lacunas.append(nome)
        else:
            grupos_map.setdefault(chave, []).append((nome, resposta))

    grupos: list[GrupoConsenso] = []
    for chave, itens in grupos_map.items():
        docs = tuple(nome for nome, _ in itens)
        valor_label = itens[0][1].strip()  # primeira resposta como rótulo
        numeros = tuple(chave[1]) if chave[0] == "num" else ()
        grupos.append(GrupoConsenso(valor=valor_label, documentos=docs, numeros=numeros))

    # Ordena do grupo mais populoso para o menor (estável por valor)
    grupos.sort(key=lambda g: (-len(g.documentos), g.valor))
    grupos_t = tuple(grupos)
    lacunas_t = tuple(lacunas)

    # Veredito agregado
    n_grupos = len(grupos_t)
    if n_grupos == 0:
        veredito = LACUNA_EVIDENCIA
        resumo = "Nenhum documento contém a informação auditada."
    elif n_grupos == 1 and not lacunas_t:
        veredito = CONSENSO
        resumo = f"Todos os {len(grupos_t[0].documentos)} documentos concordam: {grupos_t[0].valor}."
    elif n_grupos == 1 and lacunas_t:
        veredito = DIVERGENCIA
        resumo = (f"{len(grupos_t[0].documentos)} documento(s) concordam em "
                  f"'{grupos_t[0].valor}', mas {len(lacunas_t)} não contêm o dado.")
    else:
        veredito = DIVERGENCIA
        resumo = f"Divergência: {n_grupos} valores distintos entre os documentos."
        if lacunas_t:
            resumo += f" Além de {len(lacunas_t)} sem o dado."

    return ResultadoConsenso(veredito=veredito, grupos=grupos_t,
                             lacunas=lacunas_t, resumo=resumo)
