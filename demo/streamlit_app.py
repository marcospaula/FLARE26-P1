"""FLARE26 — Demo (offline, custo zero).

Vitrine read-only do benchmark e dos resultados PRÉ-COMPUTADOS já versionados no
repositório. NÃO chama nenhum LLM (não usa OpenAI, não gasta API), então pode
rodar publicamente sem custo nem risco de abuso.

Fontes (no repo):
  ../eval/gold_dataset.json   — pares (pergunta, documento) verificados
  ../eval/sample_pool.json    — respostas do FLARE por item em N execuções
  ../eval/results_curve.csv   — curva bootstrap (Pitfall A)
  ../eval/results_grid.csv    — varredura (k, t) da política de voto
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"
ABSTAIN = "ABSTAIN"
NAO_LOCALIZADO = "NÃO LOCALIZADO"
DOI_URL = "https://doi.org/10.5281/zenodo.20881699"
REPO_URL = "https://github.com/marcospaula/FLARE26-P1"


@st.cache_data
def carregar():
    gold = json.loads((EVAL / "gold_dataset.json").read_text(encoding="utf-8"))["itens"]
    pool = json.loads((EVAL / "sample_pool.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(EVAL / "results_curve.csv")
    grid = pd.read_csv(EVAL / "results_grid.csv")
    return gold, pool, curve, grid


def dominio(item: dict) -> str:
    return "Engineering (EN)" if str(item["id"]).startswith("eng") else "Legal (PT)"


st.set_page_config(page_title="FLARE26 — Glass-box Auditor (demo)", page_icon="🔎", layout="wide")
gold, pool, curve, grid = carregar()

st.title("🔎 FLARE26 — auditor de documentos que sabe dizer *“não sei”*")
st.markdown(
    "Compara documentos e, quando a evidência não corresponde ao **tipo** e ao "
    "**escopo** da pergunta, **se abstém** (*lacuna de evidência*) em vez de "
    "inventar uma resposta — evitando a **divergência falso-positiva**, o erro "
    "mais caro numa auditoria."
)
st.info(
    "🧊 **Demo offline (custo zero).** Mostra resultados **pré-computados** do "
    "benchmark publicado. Nenhum LLM é chamado aqui. "
    f"📄 [Preprint (DOI)]({DOI_URL}) · 💻 [Código + benchmark]({REPO_URL})",
    icon="🧊",
)

# ---------------------------------------------------------------- os 3 pitfalls
with st.expander("📌 A contribuição: 3 armadilhas ao *avaliar* abstenção"):
    st.markdown(
        "- **A — A ilusão do “0%”.** Uma taxa de erro rara lê-se como zero até "
        "você fazer *bootstrap*; a real era ~1–2%.\n"
        "- **B — “Difícil” é do modelo, não da taxonomia.** Conceitos vizinhos "
        "(MTBF vs. MTTR) não eram confundidos pelo modelo. Dificuldade se mede, "
        "não se presume.\n"
        "- **C — Palavra ausente ≠ conceito ausente.** Rotular “ausente” pela "
        "falta de uma palavra-chave falha em texto que parafraseia (juros → "
        "“compensação financeira”).\n\n"
        "_Cada armadilha foi um erro cometido e corrigido — as correções são a "
        "contribuição. (paper §4)_"
    )

st.divider()

# --------------------------------------------------------- explorar o benchmark
st.header("1 · Explore o benchmark (caixa de vidro)")

c1, c2 = st.columns(2)
with c1:
    dom = st.selectbox("Domínio", ["Todos", "Legal (PT)", "Engineering (EN)"])
with c2:
    tipo = st.selectbox("Tipo de item", ["Todos", "ABSTAIN (ausência)", "Com resposta"])

def passa(it: dict) -> bool:
    if dom != "Todos" and dominio(it) != dom:
        return False
    if tipo == "ABSTAIN (ausência)" and it["gold"] != ABSTAIN:
        return False
    if tipo == "Com resposta" and it["gold"] == ABSTAIN:
        return False
    return True

itens = [i for i in gold if passa(i)]
st.caption(f"{len(itens)} itens neste filtro (de {len(gold)} no total).")

if itens:
    rotulo = {i["id"]: f'{i["id"]}  —  {i["pergunta"]}' for i in itens}
    escolha = st.selectbox("Item", [i["id"] for i in itens], format_func=lambda x: rotulo[x])
    it = next(i for i in gold if i["id"] == escolha)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(f"**Pergunta:** {it['pergunta']}")
        st.markdown(f"**Documento:** `{it['documento']}`")
        if it["gold"] == ABSTAIN:
            st.markdown("**Gabarito:** :orange[**ABSTAIN**] (o documento não contém o dado)")
        else:
            st.markdown(f"**Gabarito:** :green[{it['gold']}]")
        st.markdown(f"**Por quê / evidência:** {it.get('evidencia') or '—'}")
        if not it.get("verificado", True):
            st.caption("⚠️ caso ambíguo, retido (held-out) — ver paper.")

    with col_b:
        runs = pool.get(it["id"])
        if runs:
            n = len(runs)
            cont = Counter("ABSTEVE" if r == NAO_LOCALIZADO else r for r in runs)
            abst = cont.pop("ABSTEVE", 0)
            st.markdown(f"**O que o FLARE fez em {n} execuções:**")
            if abst:
                st.markdown(f"- 🟠 absteve-se (*lacuna de evidência*): **{abst}/{n}**")
            for val, q in cont.most_common():
                st.markdown(f"- 🟢 respondeu `{val}`: **{q}/{n}**")
            if it["gold"] == ABSTAIN and abst == n:
                st.success("Abstenção **estável**: não vazou em nenhuma execução. ✅")
        else:
            st.caption("(execuções ao vivo não cacheadas para este item)")

st.divider()

# ----------------------------------------------------- resultado + pitfall A
st.header("2 · O resultado-manchete e a armadilha A")

st.markdown(
    "Extrator `gpt-4o-mini`, temperatura 0, seed fixo. Métrica-manchete = "
    "**taxa de falso-positivo** (responder quando o gabarito era `ABSTAIN`)."
)
headline = pd.DataFrame(
    {
        "Sistema": [
            "Baseline (extração livre)",
            "FLARE26 (gated)",
            "FLARE26 + voto (k=10, t=4)",
        ],
        "Falso-positivo ↓": ["38%", "~1–2%", "~0%"],
        "Recall de resposta ↑": ["100%", "81%", "80%"],
    }
)
st.table(headline)
st.caption(
    "O gate corta a divergência falso-positiva ~20–30× no conjunto sintético "
    "(30 pares). Detalhes e ressalvas no preprint."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Curva bootstrap (Pitfall A):** evento raro só aparece ao reamostrar.")
    dfc = curve.set_index("k")[["fp_mean", "recall_mean"]].rename(
        columns={"fp_mean": "falso-positivo", "recall_mean": "recall"}
    )
    st.line_chart(dfc)
    st.caption("Voto “≥1 de k” AMPLIFICA o falso-positivo conforme k cresce.")

with col2:
    st.markdown("**Varredura do limiar de voto (k, t):** maioria filtra o vazamento raro.")
    piv = grid.pivot(index="k", columns="t", values="fp").round(3)
    piv.columns = [f"t={c}" for c in piv.columns]
    st.dataframe(piv.style.format("{:.1%}", na_rep="—"))
    st.caption("Falso-positivo por (k, t). Subir t (exigir maioria) leva o FP a ~0.")

st.divider()
st.caption(
    "Benchmark pequeno e parcialmente sintético; números absolutos são "
    "indicativos. Trabalho aplicado / de avaliação — não é método novo de ML. "
    f"[Preprint]({DOI_URL}) · [Repositório]({REPO_URL}) · MIT + CC BY 4.0."
)
