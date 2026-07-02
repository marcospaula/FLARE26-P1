"""
run_eval.py — Harness de avaliação da abstenção ontológica.

Roda, sobre o gold_dataset.json, dois sistemas:
  * BASELINE: extração livre (RAG sem gating ontológico).
  * FLARE26 : extração ontologicamente restrita (pipeline.extrair_dado).

E mede a métrica-manchete: **taxa de divergência falso-positiva** —
operacionalizada aqui como "respondeu quando o gold era ABSTAIN" (alucinação).

Uso:
    OPENAI_API_KEY=... python eval/run_eval.py
(headless; importa flare26_pipeline, não usa Streamlit)
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flare26_pipeline as pipeline

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "documentos"
GOLD = Path(__file__).resolve().parent / "gold_dataset.json"

ABSTAIN = "ABSTAIN"
SENTINELAS_ABSTENCAO = {"NÃO LOCALIZADO", "NAO LOCALIZADO", "NÃO INFORMADO",
                        "NAO INFORMADO", "", "N/A", "LACUNA DE EVIDÊNCIA"}


def eh_abstencao(resposta: str) -> bool:
    return (resposta or "").strip().upper() in {s.upper() for s in SENTINELAS_ABSTENCAO}


def extrair_baseline(client, texto: str, pergunta: str) -> str | None:
    """Baseline SEM gating ontológico: responde com o que achar no contexto.
    Retorna "" para abstenção genuína (nada no texto) e None para ERRO DE
    INFRAESTRUTURA (API/parse) — que não deve ser contado como abstenção."""
    if not texto.strip():
        return ""
    prompt = (
        "Extraia do TEXTO a resposta para a PERGUNTA. Responda apenas com JSON "
        '{"resposta": "..."}. Se realmente não houver, use "".\n'
        f"PERGUNTA: {pergunta}\nTEXTO:\n{texto[:90000]}"
    )
    try:
        r = client.chat.completions.create(
            model=pipeline.MODELO_EXTRACAO,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return str(json.loads(r.choices[0].message.content).get("resposta", "")).strip()
    except Exception:
        return None


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    itens = [i for i in gold["itens"] if i.get("verificado")]

    db = os.path.join(tempfile.gettempdir(), "_flare_eval_docstore.db")
    chroma = os.path.join(tempfile.gettempdir(), "_flare_eval_chroma")
    if os.path.exists(db):
        os.remove(db)
    pipeline.iniciar_banco_sqlite(db)
    vs, _ = pipeline.criar_vector_store(persist_directory=chroma)
    client = pipeline.criar_client_openai()

    # Indexa cada documento único uma vez
    hashes: dict[str, str] = {}
    for nome in {i["documento"] for i in itens}:
        caminho = DOCS / nome
        hashes[nome] = pipeline.processar_pdf_bytes(
            caminho.read_bytes(), nome, db_path=db, vector_store=vs)

    # Acumuladores por sistema
    stats = {s: {"fp": 0, "abst_ok": 0, "ans_ok": 0, "erro_abst": 0, "erro_ans": 0}
             for s in ("BASELINE", "FLARE26")}
    n_abstain = sum(1 for i in itens if i["gold"] == ABSTAIN)
    n_answer = len(itens) - n_abstain

    print(f"\n{'id':12} {'gold':9} {'BASELINE':28} {'FLARE26':28}")
    print("-" * 80)
    for it in itens:
        fh = hashes[it["documento"]]
        ctx, _, _ = pipeline.recuperar_contexto(it["pergunta"], fh, db_path=db, vector_store=vs)

        base = extrair_baseline(client, ctx, it["pergunta"])
        base_erro = base is None
        # FLARE_CONSENSO=k usa self-consistency (votação) em vez de amostra única.
        _k = int(os.environ.get("FLARE_CONSENSO", "0"))
        if _k > 0:
            flare_ext = pipeline.extrair_dado_consenso(client, ctx, it["pergunta"], k=_k)
        else:
            flare_ext = pipeline.extrair_dado(client, ctx, it["pergunta"])
        flare, flare_erro = flare_ext.resposta_direta, flare_ext.erro_infra

        gold_abstain = it["gold"] == ABSTAIN
        for nome_sys, pred, erro in (("BASELINE", base, base_erro),
                                     ("FLARE26", flare, flare_erro)):
            if erro:
                # Erro de infra ≠ abstenção: fica fora do numerador E do denominador.
                stats[nome_sys]["erro_abst" if gold_abstain else "erro_ans"] += 1
                continue
            pred_abstain = eh_abstencao(pred)
            if gold_abstain:
                if pred_abstain:
                    stats[nome_sys]["abst_ok"] += 1
                else:
                    stats[nome_sys]["fp"] += 1  # respondeu onde devia abster = alucinação
            else:
                if not pred_abstain:
                    stats[nome_sys]["ans_ok"] += 1

        marca = "ABSTAIN" if gold_abstain else "tem-resp"
        base_disp = "ERRO" if base_erro else (base or "∅")
        flare_disp = "ERRO" if flare_erro else (flare or "∅")
        print(f"{it['id']:12} {marca:9} {base_disp[:26]:28} {flare_disp[:26]:28}")

    print("\n" + "=" * 60)
    print(f"Itens: {len(itens)} | gold ABSTAIN: {n_abstain} | gold com resposta: {n_answer}")
    print("=" * 60)
    for s in ("BASELINE", "FLARE26"):
        st = stats[s]
        # Denominadores excluem itens que erraram por infra (não são decisões).
        den_abst = n_abstain - st["erro_abst"]
        den_ans = n_answer - st["erro_ans"]
        fp_rate = st["fp"] / den_abst if den_abst else 0.0
        abst_rec = st["abst_ok"] / den_abst if den_abst else 0.0
        ans_rec = st["ans_ok"] / den_ans if den_ans else 0.0
        print(f"\n[{s}]")
        print(f"  ★ Taxa de falso-positivo (alucinou em ABSTAIN): {fp_rate:.0%} ({st['fp']}/{den_abst})")
        print(f"  Recall de abstenção (acertou o ABSTAIN):        {abst_rec:.0%}")
        print(f"  Recall de resposta (achou quando existia):      {ans_rec:.0%}")
        if st["erro_abst"] or st["erro_ans"]:
            print(f"  ⚠ Erros de infra excluídos: {st['erro_abst']} em ABSTAIN, {st['erro_ans']} em com-resposta")


if __name__ == "__main__":
    main()
