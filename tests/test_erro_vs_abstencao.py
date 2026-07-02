"""Testes do sentinela ERRO DE INFRAESTRUTURA vs. ABSTENÇÃO ontológica (F2).

Um erro de API/parse NÃO é uma decisão do gate; deve ser distinguível de uma
abstenção legítima para não contaminar a métrica de falso-positivo.
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flare26_pipeline as pipeline

_VALIDO = {
    "natureza_da_pergunta": "Monetário", "escopo_da_pergunta": "multa",
    "natureza_do_texto_encontrado": "Monetário", "escopo_do_texto_encontrado": "multa",
    "houve_compatibilidade_ontologica": True, "violou_restricao_do_usuario": False,
    "resposta_direta": "R$ 5.000", "tipo_dado": "multa", "condicionantes": "-",
    "trecho_literal": "multa de R$ 5.000", "confiabilidade": 0.9,
}


def _resp(args_dict):
    fn = types.SimpleNamespace(arguments=json.dumps(args_dict))
    tc = types.SimpleNamespace(function=fn)
    msg = types.SimpleNamespace(tool_calls=[tc])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _client(create_fn):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create_fn)))


def client_ok():
    return _client(lambda **kw: _resp(_VALIDO))


def client_erro():
    def _boom(**kw):
        raise RuntimeError("falha simulada de API")
    return _client(_boom)


def test_erro_de_infra_marca_erro_infra():
    r = pipeline.extrair_dado(client_erro(), "algum texto", "qual a multa?")
    assert r.erro_infra is True
    assert r.resposta_direta == "NÃO LOCALIZADO"  # mesma string de abstenção...


def test_abstencao_genuina_nao_marca_erro():
    assert pipeline.resultado_vazio().erro_infra is False
    # texto vazio => abstenção legítima, sem chamar o cliente
    r = pipeline.extrair_dado(client_ok(), "", "qual a multa?")
    assert r.erro_infra is False


def test_extracao_valida_nao_marca_erro():
    r = pipeline.extrair_dado(client_ok(), "multa de R$ 5.000", "qual a multa?")
    assert r.erro_infra is False
    assert r.resposta_direta == "R$ 5.000"


def test_schema_enviado_ao_llm_nao_expoe_erro_infra():
    props = pipeline._SCHEMA_LLM.get("properties", {})
    assert "erro_infra" not in props
    assert "erro_infra" not in pipeline._SCHEMA_LLM.get("required", [])
    # mas o campo existe no modelo (uso interno)
    assert "erro_infra" in pipeline.ExtracaoUniversal.model_fields


def test_consenso_todas_amostras_com_erro_retorna_erro():
    r = pipeline.extrair_dado_consenso(client_erro(), "texto", "q?", k=3)
    assert r.erro_infra is True


def test_consenso_ignora_erros_e_vota_nas_validas():
    estado = {"n": 0}

    def _create(**kw):
        estado["n"] += 1
        if estado["n"] == 1:
            raise RuntimeError("erro na 1a amostra")
        return _resp(_VALIDO)

    r = pipeline.extrair_dado_consenso(_client(_create), "texto", "q?", k=3)
    assert r.erro_infra is False
    assert r.resposta_direta == "R$ 5.000"
