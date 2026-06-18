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


def extrair_baseline(client, texto: str, pergunta: str) -> str:
    """Baseline SEM gating ontológico: responde com o que achar no contexto."""
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
        return ""


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
    stats = {s: {"fp": 0, "abst_ok": 0, "ans_ok": 0} for s in ("BASELINE", "FLARE26")}
    n_abstain = sum(1 for i in itens if i["gold"] == ABSTAIN)
    n_answer = len(itens) - n_abstain

    print(f"\n{'id':12} {'gold':9} {'BASELINE':28} {'FLARE26':28}")
    print("-" * 80)
    for it in itens:
        fh = hashes[it["documento"]]
        ctx, _, _ = pipeline.recuperar_contexto(it["pergunta"], fh, db_path=db, vector_store=vs)

        base = extrair_baseline(client, ctx, it["pergunta"])
        # FLARE_CONSENSO=k usa self-consistency (votação) em vez de amostra única.
        _k = int(os.environ.get("FLARE_CONSENSO", "0"))
        if _k > 0:
            flare = pipeline.extrair_dado_consenso(client, ctx, it["pergunta"], k=_k).resposta_direta
        else:
            flare = pipeline.extrair_dado(client, ctx, it["pergunta"]).resposta_direta

        gold_abstain = it["gold"] == ABSTAIN
        for nome_sys, pred in (("BASELINE", base), ("FLARE26", flare)):
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
        print(f"{it['id']:12} {marca:9} {(base or '∅')[:26]:28} {(flare or '∅')[:26]:28}")

    print("\n" + "=" * 60)
    print(f"Itens: {len(itens)} | gold ABSTAIN: {n_abstain} | gold com resposta: {n_answer}")
    print("=" * 60)
    for s in ("BASELINE", "FLARE26"):
        st = stats[s]
        fp_rate = st["fp"] / n_abstain if n_abstain else 0.0
        abst_rec = st["abst_ok"] / n_abstain if n_abstain else 0.0
        ans_rec = st["ans_ok"] / n_answer if n_answer else 0.0
        print(f"\n[{s}]")
        print(f"  ★ Taxa de falso-positivo (alucinou em ABSTAIN): {fp_rate:.0%} ({st['fp']}/{n_abstain})")
        print(f"  Recall de abstenção (acertou o ABSTAIN):        {abst_rec:.0%}")
        print(f"  Recall de resposta (achou quando existia):      {ans_rec:.0%}")


if __name__ == "__main__":
    main()
