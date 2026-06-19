"""
run_eval_agg.py — Benchmark AGREGADO (média ± desvio) para o draft do paper.

Coleta UM pool de amostras por item (caro: chamadas de LLM) e então deriva
estatísticas por reamostragem, sem novas chamadas:
  * BASELINE: média ± desvio sobre POOL_BASE execuções.
  * FLARE26 (estrito): média ± desvio sobre POOL_FLARE execuções.
  * FLARE26 + self-consistency (k): bootstrap — sorteia k amostras do pool
    B_BOOT vezes e agrega ("responder se >=1 responde"), dando média ± desvio.

Uso: OPENAI_API_KEY=... python eval/run_eval_agg.py
Env: POOL_FLARE (default 8), POOL_BASE (3), K_CONS (5), B_BOOT (300).
"""

from __future__ import annotations

import os
import sys
import json
import random
import tempfile
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flare26_pipeline as pipeline

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "documentos"
GOLD = Path(__file__).resolve().parent / "gold_dataset.json"

POOL_FLARE = int(os.environ.get("POOL_FLARE", "8"))
POOL_BASE = int(os.environ.get("POOL_BASE", "3"))
K_CONS = int(os.environ.get("K_CONS", "5"))
B_BOOT = int(os.environ.get("B_BOOT", "300"))

ABSTAIN = "ABSTAIN"
SENTINELAS = {"NÃO LOCALIZADO", "NAO LOCALIZADO", "NÃO INFORMADO", "NAO INFORMADO",
              "", "N/A", "LACUNA DE EVIDÊNCIA"}


def eh_abstencao(resp: str) -> bool:
    return (resp or "").strip().upper() in {s.upper() for s in SENTINELAS}


def extrair_baseline(client, texto, pergunta):
    if not texto.strip():
        return ""
    prompt = ('Extraia do TEXTO a resposta para a PERGUNTA. Responda apenas JSON '
              '{"resposta": "..."}. Se realmente não houver, use "".\n'
              f"PERGUNTA: {pergunta}\nTEXTO:\n{texto[:90000]}")
    try:
        r = client.chat.completions.create(model=pipeline.MODELO_EXTRACAO,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return str(json.loads(r.choices[0].message.content).get("resposta", "")).strip()
    except Exception:
        return ""


def metricas(respondeu_por_item, itens):
    ab = [i for i in itens if i["gold"] == ABSTAIN]
    resp = [i for i in itens if i["gold"] != ABSTAIN]
    fp = sum(1 for i in ab if respondeu_por_item[i["id"]]) / len(ab)
    ar = sum(1 for i in ab if not respondeu_por_item[i["id"]]) / len(ab)
    rr = sum(1 for i in resp if respondeu_por_item[i["id"]]) / len(resp)
    return fp, ar, rr


def ms(valores):
    return statistics.mean(valores), (statistics.pstdev(valores) if len(valores) > 1 else 0.0)


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    itens = [i for i in gold["itens"] if i.get("verificado")]

    db = os.path.join(tempfile.gettempdir(), "_agg_docstore.db")
    if os.path.exists(db):
        os.remove(db)
    pipeline.iniciar_banco_sqlite(db)
    vs, _ = pipeline.criar_vector_store(persist_directory=os.path.join(tempfile.gettempdir(), "_agg_chroma"))
    client = pipeline.criar_client_openai()

    hashes = {n: pipeline.processar_pdf_bytes((DOCS / n).read_bytes(), n, db_path=db, vector_store=vs)
              for n in {i["documento"] for i in itens}}

    # Coleta pools (única fase com custo de API)
    pool_flare, pool_base = {}, {}
    n_calls = len(itens) * (POOL_FLARE + POOL_BASE)
    print(f"Coletando pools: {len(itens)} itens x ({POOL_FLARE} FLARE + {POOL_BASE} base) = {n_calls} chamadas...")
    for it in itens:
        ctx, _, _ = pipeline.recuperar_contexto(it["pergunta"], hashes[it["documento"]], db_path=db, vector_store=vs)
        pool_flare[it["id"]] = [pipeline.extrair_dado(client, ctx, it["pergunta"]).resposta_direta for _ in range(POOL_FLARE)]
        pool_base[it["id"]] = [extrair_baseline(client, ctx, it["pergunta"]) for _ in range(POOL_BASE)]

    # BASELINE: média sobre POOL_BASE execuções
    base_runs = [metricas({it["id"]: not eh_abstencao(pool_base[it["id"]][r]) for it in itens}, itens)
                 for r in range(POOL_BASE)]
    # FLARE estrito: média sobre POOL_FLARE execuções
    flare_runs = [metricas({it["id"]: not eh_abstencao(pool_flare[it["id"]][r]) for it in itens}, itens)
                  for r in range(POOL_FLARE)]
    # FLARE + consenso k: bootstrap do pool
    rng = random.Random(42)
    cons_runs = []
    for _ in range(B_BOOT):
        dec = {}
        for it in itens:
            amostras = [pool_flare[it["id"]][rng.randrange(POOL_FLARE)] for _ in range(K_CONS)]
            dec[it["id"]] = any(not eh_abstencao(a) for a in amostras)  # responder se >=1 responde
        cons_runs.append(metricas(dec, itens))

    def linha(nome, runs):
        fp = ms([r[0] for r in runs]); ar = ms([r[1] for r in runs]); rr = ms([r[2] for r in runs])
        return f"  {nome:34} {fp[0]:5.0%} ± {fp[1]:3.0%}   {ar[0]:5.0%} ± {ar[1]:3.0%}   {rr[0]:5.0%} ± {rr[1]:3.0%}"

    print(f"\n{'='*78}")
    print(f"BENCHMARK AGREGADO — {len(itens)} itens (13 ABSTAIN, 17 resposta)")
    print(f"pools: FLARE={POOL_FLARE}, base={POOL_BASE}; consenso k={K_CONS}, bootstrap={B_BOOT}")
    print('='*78)
    print(f"  {'Sistema':34} {'Falso-pos':13} {'Rec.abst':13} {'Rec.resp':13}")
    print("  " + "-"*74)
    print(linha("BASELINE (extração livre)", base_runs))
    print(linha("FLARE26 (gating estrita)", flare_runs))
    print(linha(f"FLARE26 + self-consistency (k={K_CONS})", cons_runs))


if __name__ == "__main__":
    main()
