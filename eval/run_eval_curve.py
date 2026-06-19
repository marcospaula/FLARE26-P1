"""
run_eval_curve.py — Curva recall x custo da self-consistency (varredura de k).

Coleta um pool de P amostras FLARE por item (única fase com custo de API) e, por
bootstrap, estima para cada k em KS: falso-positivo e recall de resposta
(média ± desvio), com a política "responder se >=1 de k responde". O eixo de
custo é o nº de chamadas de LLM por documento (= k).

Salva os pontos em eval/results_curve.csv (dados da figura do paper).

Uso: OPENAI_API_KEY=... python eval/run_eval_curve.py
Env: POOL (default 10), B_BOOT (400), KS (ex "1,2,3,5,8,10").
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
OUT_CSV = Path(__file__).resolve().parent / "results_curve.csv"

POOL = int(os.environ.get("POOL", "10"))
B_BOOT = int(os.environ.get("B_BOOT", "400"))
KS = [int(x) for x in os.environ.get("KS", "1,2,3,5,8,10").split(",")]

ABSTAIN = "ABSTAIN"
SENTINELAS = {"NÃO LOCALIZADO", "NAO LOCALIZADO", "NÃO INFORMADO", "NAO INFORMADO",
              "", "N/A", "LACUNA DE EVIDÊNCIA"}


def eh_abstencao(resp: str) -> bool:
    return (resp or "").strip().upper() in {s.upper() for s in SENTINELAS}


def metricas(respondeu, itens):
    ab = [i for i in itens if i["gold"] == ABSTAIN]
    resp = [i for i in itens if i["gold"] != ABSTAIN]
    fp = sum(1 for i in ab if respondeu[i["id"]]) / len(ab)
    rr = sum(1 for i in resp if respondeu[i["id"]]) / len(resp)
    return fp, rr


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    itens = [i for i in gold["itens"] if i.get("verificado")]

    db = os.path.join(tempfile.gettempdir(), "_curve_docstore.db")
    if os.path.exists(db):
        os.remove(db)
    pipeline.iniciar_banco_sqlite(db)
    vs, _ = pipeline.criar_vector_store(persist_directory=os.path.join(tempfile.gettempdir(), "_curve_chroma"))
    client = pipeline.criar_client_openai()

    hashes = {n: pipeline.processar_pdf_bytes((DOCS / n).read_bytes(), n, db_path=db, vector_store=vs)
              for n in {i["documento"] for i in itens}}

    print(f"Coletando pool: {len(itens)} itens x {POOL} amostras = {len(itens)*POOL} chamadas...")
    pool = {}
    for it in itens:
        ctx, _, _ = pipeline.recuperar_contexto(it["pergunta"], hashes[it["documento"]], db_path=db, vector_store=vs)
        pool[it["id"]] = [pipeline.extrair_dado(client, ctx, it["pergunta"]).resposta_direta for _ in range(POOL)]

    rng = random.Random(42)
    linhas = []
    for k in KS:
        fps, rrs = [], []
        for _ in range(B_BOOT):
            dec = {}
            for it in itens:
                amostras = [pool[it["id"]][rng.randrange(POOL)] for _ in range(k)]
                dec[it["id"]] = any(not eh_abstencao(a) for a in amostras)
            fp, rr = metricas(dec, itens)
            fps.append(fp); rrs.append(rr)
        linhas.append((k, statistics.mean(fps), statistics.pstdev(fps),
                       statistics.mean(rrs), statistics.pstdev(rrs)))

    print(f"\n{'='*64}")
    print(f"CURVA RECALL x CUSTO — {len(itens)} itens (pool={POOL}, bootstrap={B_BOOT})")
    print('='*64)
    print(f"  {'k (chamadas/doc)':18} {'Falso-pos':14} {'Recall resposta':16}")
    print("  " + "-"*52)
    for k, fpm, fps, rrm, rrs in linhas:
        print(f"  {k:<18} {fpm:5.0%} ± {fps:3.0%}     {rrm:5.0%} ± {rrs:3.0%}")

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("k,fp_mean,fp_std,recall_mean,recall_std\n")
        for k, fpm, fps, rrm, rrs in linhas:
            f.write(f"{k},{fpm:.4f},{fps:.4f},{rrm:.4f},{rrs:.4f}\n")
    print(f"\nDados salvos em {OUT_CSV.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
