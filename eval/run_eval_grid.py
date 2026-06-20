"""
run_eval_grid.py — Varredura (k, t) da self-consistency, com pool CACHEADO.

Coleta um pool de P amostras FLARE por item UMA vez e salva em
eval/sample_pool.json. Análises posteriores (qualquer k, t) são offline e
gratuitas (sem novas chamadas de API).

Política de voto: "responder se >= t das k amostras responderem". A hipótese é
que t = maioria filtra vazamentos raros (que aparecem em 1-2 de k) e baixa o
falso-positivo sem perder recall das respostas estáveis.

Uso: OPENAI_API_KEY=... python eval/run_eval_grid.py        # coleta se faltar
     python eval/run_eval_grid.py --analyze                 # só análise (offline)
Env: POOL (default 10), B_BOOT (400).
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

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "documentos"
GOLD = Path(__file__).resolve().parent / "gold_dataset.json"
POOL_CACHE = Path(__file__).resolve().parent / "sample_pool.json"
OUT_CSV = Path(__file__).resolve().parent / "results_grid.csv"

POOL = int(os.environ.get("POOL", "10"))
B_BOOT = int(os.environ.get("B_BOOT", "400"))
KS = [1, 3, 5, 8, 10]

ABSTAIN = "ABSTAIN"
SENTINELAS = {"NÃO LOCALIZADO", "NAO LOCALIZADO", "NÃO INFORMADO", "NAO INFORMADO",
              "", "N/A", "LACUNA DE EVIDÊNCIA"}


def eh_abstencao(resp: str) -> bool:
    return (resp or "").strip().upper() in {s.upper() for s in SENTINELAS}


def coletar_pool(itens):
    """Coleta P amostras por item e salva em cache. Requer API."""
    import flare26_pipeline as pipeline
    db = os.path.join(tempfile.gettempdir(), "_grid_docstore.db")
    if os.path.exists(db):
        os.remove(db)
    pipeline.iniciar_banco_sqlite(db)
    vs, _ = pipeline.criar_vector_store(persist_directory=os.path.join(tempfile.gettempdir(), "_grid_chroma"))
    client = pipeline.criar_client_openai()
    hashes = {n: pipeline.processar_pdf_bytes((DOCS / n).read_bytes(), n, db_path=db, vector_store=vs)
              for n in {i["documento"] for i in itens}}
    print(f"Coletando pool: {len(itens)} x {POOL} = {len(itens)*POOL} chamadas...")
    pool = {}
    for it in itens:
        ctx, _, _ = pipeline.recuperar_contexto(it["pergunta"], hashes[it["documento"]], db_path=db, vector_store=vs)
        pool[it["id"]] = [pipeline.extrair_dado(client, ctx, it["pergunta"]).resposta_direta for _ in range(POOL)]
    POOL_CACHE.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Pool salvo em {POOL_CACHE.relative_to(RAIZ)}")
    return pool


def metricas(respondeu, itens):
    ab = [i for i in itens if i["gold"] == ABSTAIN]
    resp = [i for i in itens if i["gold"] != ABSTAIN]
    fp = sum(1 for i in ab if respondeu[i["id"]]) / len(ab)
    rr = sum(1 for i in resp if respondeu[i["id"]]) / len(resp)
    return fp, rr


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    itens = [i for i in gold["itens"] if i.get("verificado")]

    if POOL_CACHE.exists():
        pool = json.loads(POOL_CACHE.read_text(encoding="utf-8"))
        print(f"Pool em cache: {POOL_CACHE.relative_to(RAIZ)} ({len(pool)} itens)")
    else:
        if "--analyze" in sys.argv:
            print("Sem cache de pool e modo --analyze; rode sem --analyze para coletar."); return
        pool = coletar_pool(itens)

    P = len(next(iter(pool.values())))
    rng = random.Random(42)

    print(f"\n{'='*70}")
    print(f"VARREDURA (k, t) — responder se >= t de k respondem  (pool={P}, boot={B_BOOT})")
    print('='*70)
    rows = []
    for k in KS:
        for t in range(1, k + 1):
            fps, rrs = [], []
            for _ in range(B_BOOT):
                dec = {}
                for it in itens:
                    amostras = [pool[it["id"]][rng.randrange(P)] for _ in range(k)]
                    n_resp = sum(1 for a in amostras if not eh_abstencao(a))
                    dec[it["id"]] = (n_resp >= t)
                fp, rr = metricas(dec, itens)
                fps.append(fp); rrs.append(rr)
            rows.append((k, t, statistics.mean(fps), statistics.mean(rrs)))

    print(f"  {'k':>3} {'t':>3} {'limiar':>10} {'FP':>8} {'Recall':>9}")
    print("  " + "-"*40)
    for k, t, fp, rr in rows:
        marca = "  <-- maioria" if t == (k // 2 + 1) else ""
        print(f"  {k:>3} {t:>3} {f'{t}/{k}':>10} {fp:>7.0%} {rr:>8.0%}{marca}")

    # destaca melhor ponto: minimiza FP exigindo recall >= 0.85
    viaveis = [r for r in rows if r[3] >= 0.85]
    if viaveis:
        best = min(viaveis, key=lambda r: (r[2], -r[3]))
        print(f"\n  Melhor ponto com recall>=85%: k={best[0]}, t={best[1]} "
              f"-> FP {best[2]:.0%}, recall {best[3]:.0%}")

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("k,t,fp,recall\n")
        for k, t, fp, rr in rows:
            f.write(f"{k},{t},{fp:.4f},{rr:.4f}\n")
    print(f"  Grade salva em {OUT_CSV.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
