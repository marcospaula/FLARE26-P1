"""
run_eval_repeat.py — Self-consistency (voto) + ruído de k execuções.

Roda o extrator FLARE26 R vezes por item, reaproveitando as amostras para:
  * PASSO 6: caracterizar o RUÍDO — métricas por execução, com média ± desvio.
  * PASSO 5: VOTAÇÃO — agregar as R amostras por item e decidir responder/abster,
    buscando recuperar recall SEM reabrir falso-positivo.

A pergunta central: existe um limiar de voto que mantém 0% de falso-positivo
e recupera parte do recall? O diagnóstico de "vazamento por item" responde isso.

Uso: OPENAI_API_KEY=... python eval/run_eval_repeat.py
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flare26_pipeline as pipeline

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "documentos"
GOLD = Path(__file__).resolve().parent / "gold_dataset.json"

R = 5  # execuções por item
ABSTAIN = "ABSTAIN"
SENTINELAS = {"NÃO LOCALIZADO", "NAO LOCALIZADO", "NÃO INFORMADO", "NAO INFORMADO",
              "", "N/A", "LACUNA DE EVIDÊNCIA"}


def eh_abstencao(resp: str) -> bool:
    return (resp or "").strip().upper() in {s.upper() for s in SENTINELAS}


def metricas(decisao_por_item, itens):
    """decisao_por_item: dict id->bool (True=respondeu). Retorna (fp_rate, abst_rec, ans_rec)."""
    ab = [i for i in itens if i["gold"] == ABSTAIN]
    resp = [i for i in itens if i["gold"] != ABSTAIN]
    fp = sum(1 for i in ab if decisao_por_item[i["id"]])
    abst_ok = sum(1 for i in ab if not decisao_por_item[i["id"]])
    ans_ok = sum(1 for i in resp if decisao_por_item[i["id"]])
    return (fp / len(ab), abst_ok / len(ab), ans_ok / len(resp))


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    itens = [i for i in gold["itens"] if i.get("verificado")]

    db = os.path.join(tempfile.gettempdir(), "_rep_docstore.db")
    if os.path.exists(db):
        os.remove(db)
    pipeline.iniciar_banco_sqlite(db)
    vs, _ = pipeline.criar_vector_store(persist_directory=os.path.join(tempfile.gettempdir(), "_rep_chroma"))
    client = pipeline.criar_client_openai()

    hashes = {}
    for nome in {i["documento"] for i in itens}:
        hashes[nome] = pipeline.processar_pdf_bytes((DOCS / nome).read_bytes(), nome, db_path=db, vector_store=vs)

    # Coleta R respostas por item (e o contexto reaproveitado)
    respostas = {}  # id -> list[str] de tamanho R
    for it in itens:
        ctx, _, _ = pipeline.recuperar_contexto(it["pergunta"], hashes[it["documento"]], db_path=db, vector_store=vs)
        respostas[it["id"]] = [pipeline.extrair_dado(client, ctx, it["pergunta"]).resposta_direta for _ in range(R)]

    # contagem de "respondeu" por item
    n_resp = {iid: sum(0 if eh_abstencao(r) else 1 for r in lst) for iid, lst in respostas.items()}

    # ---- PASSO 6: ruído por execução ----
    por_run = [metricas({it["id"]: not eh_abstencao(respostas[it["id"]][r]) for it in itens}, itens) for r in range(R)]
    def ms(idx):
        vals = [m[idx] for m in por_run]
        return statistics.mean(vals), (statistics.pstdev(vals))

    print(f"\n===== PASSO 6: RUÍDO ENTRE {R} EXECUÇÕES (FLARE26 estrito) =====")
    for nome, idx in [("Falso-positivo", 0), ("Recall abstenção", 1), ("Recall resposta", 2)]:
        media, dp = ms(idx)
        print(f"  {nome:18}: {media:.0%}  (± {dp:.0%})")

    # ---- diagnóstico: vazamento por item ABSTAIN ----
    print(f"\n===== VAZAMENTO POR ITEM (quantas de {R} execuções RESPONDERAM) =====")
    print("  ABSTAIN (queremos 0):")
    for it in itens:
        if it["gold"] == ABSTAIN and n_resp[it["id"]] > 0:
            print(f"    {it['id']:22} respondeu {n_resp[it['id']]}/{R}  ⚠️")
    abst_estaveis = sum(1 for it in itens if it["gold"] == ABSTAIN and n_resp[it["id"]] == 0)
    print(f"  -> {abst_estaveis}/{sum(1 for i in itens if i['gold']==ABSTAIN)} ABSTAIN ficaram 100% estáveis (0 vazamento)")
    print("  COM RESPOSTA instáveis (oscilaram):")
    for it in itens:
        if it["gold"] != ABSTAIN and 0 < n_resp[it["id"]] < R:
            print(f"    {it['id']:22} respondeu {n_resp[it['id']]}/{R}  (flaky)")

    # ---- PASSO 5: votação por limiar ----
    print(f"\n===== PASSO 5: VOTAÇÃO (responder se >= t de {R} execuções responderem) =====")
    print(f"  {'limiar t':10} {'Falso-pos':10} {'Rec.abst':10} {'Rec.resp':10}")
    for t in range(1, R + 1):
        dec = {it["id"]: (n_resp[it["id"]] >= t) for it in itens}
        fp, ar, rr = metricas(dec, itens)
        print(f"  >= {t:<7} {fp:<10.0%} {ar:<10.0%} {rr:<10.0%}")


if __name__ == "__main__":
    main()
