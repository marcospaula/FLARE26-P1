"""FLARE26 — Demo (offline, zero-cost / custo zero).

Read-only showcase of the benchmark and the PRE-COMPUTED results versioned in the
repo. It does NOT call any LLM (no OpenAI, no API spend), so it can run publicly
with no cost and no abuse risk. UI is bilingual (EN/PT); benchmark items stay in
their source language (legal = PT, engineering = EN).

Sources (in repo):
  ../eval/gold_dataset.json   — verified (question, document) pairs
  ../eval/sample_pool.json    — FLARE answers per item over N runs
  ../eval/results_curve.csv   — bootstrap curve (Pitfall A)
  ../eval/results_grid.csv    — (k, t) vote-threshold sweep
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

STR = {
    "en": {
        "title": "🔎 FLARE26 — a document auditor that knows when to say *“I don't know”*",
        "subtitle": (
            "It compares documents and, when the evidence doesn't match the "
            "question's **type** and **scope**, it **abstains** (*evidence gap*) "
            "instead of fabricating an answer — avoiding a **false-positive "
            "divergence**, the costliest error in an audit."
        ),
        "info": (
            "🧊 **Offline demo (zero-cost).** Shows **pre-computed** results from "
            f"the published benchmark. No LLM is called here. 📄 [Preprint (DOI)]({DOI_URL})"
            f" · 💻 [Code + benchmark]({REPO_URL})"
        ),
        "pitfalls_title": "📌 The contribution: 3 pitfalls in *evaluating* abstention",
        "pitfalls_body": (
            "- **A — The “0%” illusion.** A rare error rate reads as zero until you "
            "*bootstrap*; the real one was ~1–2%.\n"
            "- **B — “Hard” is the model's, not the taxonomy's.** Near-neighbor "
            "concepts (MTBF vs. MTTR) weren't confused by the model. Difficulty is "
            "measured, not assumed.\n"
            "- **C — Missing keyword ≠ missing concept.** Labeling “absent” by a "
            "missing keyword fails on paraphrasing text (interest → “financial "
            "compensation”).\n\n"
            "_Each pitfall was a mistake made and corrected — the corrections are the "
            "contribution. (paper §4)_"
        ),
        "h1": "1 · Explore the benchmark (glass-box)",
        "domain": "Domain",
        "domain_opts": ["All", "Legal (PT)", "Engineering (EN)"],
        "kind": "Item type",
        "kind_opts": ["All", "ABSTAIN (absence)", "Answerable"],
        "count": "{n} items in this filter (of {tot} total).",
        "item": "Item",
        "lbl_q": "**Question:**",
        "lbl_doc": "**Document:**",
        "lbl_gold_abstain": "**Gold:** :orange[**ABSTAIN**] (the document does not contain the value)",
        "lbl_gold": "**Gold:** :green[{v}]",
        "lbl_why": "**Why / evidence:**",
        "held_out": "⚠️ ambiguous case, held out — see paper.",
        "runs_title": "**What FLARE did over {n} runs:**",
        "abstained": "- 🟠 abstained (*evidence gap*): **{a}/{n}**",
        "answered": "- 🟢 answered `{v}`: **{q}/{n}**",
        "stable": "Abstention is **stable**: it never leaked across runs. ✅",
        "no_pool": "(live runs not cached for this item)",
        "h2": "2 · The headline result and Pitfall A",
        "extractor": (
            "Extractor `gpt-4o-mini`, temperature 0, fixed seed. Headline metric = "
            "**false-positive rate** (answering when the gold was `ABSTAIN`)."
        ),
        "sys": ["Baseline (free extraction)", "FLARE26 (gated)", "FLARE26 + vote (k=10, t=4)"],
        "col_sys": "System",
        "col_fp": "False-positive ↓",
        "col_rec": "Answer recall ↑",
        "headline_cap": (
            "The gate cuts false-positive divergence ~20–30× on the synthetic set "
            "(30 pairs). Details and caveats in the preprint."
        ),
        "curve_title": "**Bootstrap curve (Pitfall A):** a rare event only shows up when you resample.",
        "curve_cap": "A “≥1 of k” vote AMPLIFIES the false-positive as k grows.",
        "grid_title": "**Vote-threshold sweep (k, t):** a majority filters the rare leak.",
        "grid_cap": "False-positive by (k, t). Raising t (require a majority) drives FP to ~0.",
        "footer": (
            "Small, partly synthetic benchmark; absolute numbers are indicative. "
            "Applied / evaluation work — not a new ML method. "
            f"[Preprint]({DOI_URL}) · [Repo]({REPO_URL}) · MIT + CC BY 4.0."
        ),
        "fp_series": "false-positive",
        "rec_series": "recall",
        "data_note": "ℹ️ Benchmark items appear in their source language (legal = PT, engineering = EN).",
    },
    "pt": {
        "title": "🔎 FLARE26 — auditor de documentos que sabe dizer *“não sei”*",
        "subtitle": (
            "Compara documentos e, quando a evidência não corresponde ao **tipo** "
            "e ao **escopo** da pergunta, **se abstém** (*lacuna de evidência*) em "
            "vez de inventar uma resposta — evitando a **divergência falso-positiva**, "
            "o erro mais caro numa auditoria."
        ),
        "info": (
            "🧊 **Demo offline (custo zero).** Mostra resultados **pré-computados** do "
            f"benchmark publicado. Nenhum LLM é chamado aqui. 📄 [Preprint (DOI)]({DOI_URL})"
            f" · 💻 [Código + benchmark]({REPO_URL})"
        ),
        "pitfalls_title": "📌 A contribuição: 3 armadilhas ao *avaliar* abstenção",
        "pitfalls_body": (
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
        ),
        "h1": "1 · Explore o benchmark (caixa de vidro)",
        "domain": "Domínio",
        "domain_opts": ["Todos", "Jurídico (PT)", "Engenharia (EN)"],
        "kind": "Tipo de item",
        "kind_opts": ["Todos", "ABSTAIN (ausência)", "Com resposta"],
        "count": "{n} itens neste filtro (de {tot} no total).",
        "item": "Item",
        "lbl_q": "**Pergunta:**",
        "lbl_doc": "**Documento:**",
        "lbl_gold_abstain": "**Gabarito:** :orange[**ABSTAIN**] (o documento não contém o dado)",
        "lbl_gold": "**Gabarito:** :green[{v}]",
        "lbl_why": "**Por quê / evidência:**",
        "held_out": "⚠️ caso ambíguo, retido (held-out) — ver paper.",
        "runs_title": "**O que o FLARE fez em {n} execuções:**",
        "abstained": "- 🟠 absteve-se (*lacuna de evidência*): **{a}/{n}**",
        "answered": "- 🟢 respondeu `{v}`: **{q}/{n}**",
        "stable": "Abstenção **estável**: não vazou em nenhuma execução. ✅",
        "no_pool": "(execuções ao vivo não cacheadas para este item)",
        "h2": "2 · O resultado-manchete e a armadilha A",
        "extractor": (
            "Extrator `gpt-4o-mini`, temperatura 0, seed fixo. Métrica-manchete = "
            "**taxa de falso-positivo** (responder quando o gabarito era `ABSTAIN`)."
        ),
        "sys": ["Baseline (extração livre)", "FLARE26 (gated)", "FLARE26 + voto (k=10, t=4)"],
        "col_sys": "Sistema",
        "col_fp": "Falso-positivo ↓",
        "col_rec": "Recall de resposta ↑",
        "headline_cap": (
            "O gate corta a divergência falso-positiva ~20–30× no conjunto sintético "
            "(30 pares). Detalhes e ressalvas no preprint."
        ),
        "curve_title": "**Curva bootstrap (Pitfall A):** evento raro só aparece ao reamostrar.",
        "curve_cap": "Voto “≥1 de k” AMPLIFICA o falso-positivo conforme k cresce.",
        "grid_title": "**Varredura do limiar de voto (k, t):** maioria filtra o vazamento raro.",
        "grid_cap": "Falso-positivo por (k, t). Subir t (exigir maioria) leva o FP a ~0.",
        "footer": (
            "Benchmark pequeno e parcialmente sintético; números absolutos são "
            "indicativos. Trabalho aplicado / de avaliação — não é método novo de ML. "
            f"[Preprint]({DOI_URL}) · [Repositório]({REPO_URL}) · MIT + CC BY 4.0."
        ),
        "fp_series": "falso-positivo",
        "rec_series": "recall",
        "data_note": "ℹ️ Os itens do benchmark aparecem na língua original (jurídico = PT, engenharia = EN).",
    },
}


@st.cache_data
def carregar():
    gold = json.loads((EVAL / "gold_dataset.json").read_text(encoding="utf-8"))["itens"]
    pool = json.loads((EVAL / "sample_pool.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(EVAL / "results_curve.csv")
    grid = pd.read_csv(EVAL / "results_grid.csv")
    return gold, pool, curve, grid


def dominio(item: dict) -> str:
    return "eng" if str(item["id"]).startswith("eng") else "legal"


st.set_page_config(page_title="FLARE26 — Glass-box Auditor (demo)", page_icon="🔎", layout="wide")
gold, pool, curve, grid = carregar()

idioma = st.sidebar.radio("🌐 Language / Idioma", ["English", "Português"], index=0)
T = STR["en" if idioma == "English" else "pt"]

st.title(T["title"])
st.markdown(T["subtitle"])
st.info(T["info"], icon="🧊")

with st.expander(T["pitfalls_title"]):
    st.markdown(T["pitfalls_body"])

st.divider()

# --------------------------------------------------------- explore o benchmark
st.header(T["h1"])
st.caption(T["data_note"])

c1, c2 = st.columns(2)
with c1:
    dom = st.selectbox(T["domain"], T["domain_opts"])
with c2:
    tipo = st.selectbox(T["kind"], T["kind_opts"])

dom_idx = T["domain_opts"].index(dom)      # 0 all, 1 legal, 2 eng
kind_idx = T["kind_opts"].index(tipo)      # 0 all, 1 abstain, 2 answerable

def passa(it: dict) -> bool:
    if dom_idx == 1 and dominio(it) != "legal":
        return False
    if dom_idx == 2 and dominio(it) != "eng":
        return False
    if kind_idx == 1 and it["gold"] != ABSTAIN:
        return False
    if kind_idx == 2 and it["gold"] == ABSTAIN:
        return False
    return True

itens = [i for i in gold if passa(i)]
st.caption(T["count"].format(n=len(itens), tot=len(gold)))

if itens:
    rotulo = {i["id"]: f'{i["id"]}  —  {i["pergunta"]}' for i in itens}
    escolha = st.selectbox(T["item"], [i["id"] for i in itens], format_func=lambda x: rotulo[x])
    it = next(i for i in gold if i["id"] == escolha)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(f'{T["lbl_q"]} {it["pergunta"]}')
        st.markdown(f'{T["lbl_doc"]} `{it["documento"]}`')
        if it["gold"] == ABSTAIN:
            st.markdown(T["lbl_gold_abstain"])
        else:
            st.markdown(T["lbl_gold"].format(v=it["gold"]))
        st.markdown(f'{T["lbl_why"]} {it.get("evidencia") or "—"}')
        if not it.get("verificado", True):
            st.caption(T["held_out"])

    with col_b:
        runs = pool.get(it["id"])
        if runs:
            n = len(runs)
            cont = Counter("ABSTEVE" if r == NAO_LOCALIZADO else r for r in runs)
            abst = cont.pop("ABSTEVE", 0)
            st.markdown(T["runs_title"].format(n=n))
            if abst:
                st.markdown(T["abstained"].format(a=abst, n=n))
            for val, q in cont.most_common():
                st.markdown(T["answered"].format(v=val, q=q, n=n))
            if it["gold"] == ABSTAIN and abst == n:
                st.success(T["stable"])
        else:
            st.caption(T["no_pool"])

st.divider()

# ----------------------------------------------------- resultado + pitfall A
st.header(T["h2"])
st.markdown(T["extractor"])

headline = pd.DataFrame(
    {
        T["col_sys"]: T["sys"],
        T["col_fp"]: ["38%", "~1–2%", "~0%"],
        T["col_rec"]: ["100%", "81%", "80%"],
    }
)
st.table(headline)
st.caption(T["headline_cap"])

col1, col2 = st.columns(2)
with col1:
    st.markdown(T["curve_title"])
    dfc = curve.set_index("k")[["fp_mean", "recall_mean"]].rename(
        columns={"fp_mean": T["fp_series"], "recall_mean": T["rec_series"]}
    )
    st.line_chart(dfc)
    st.caption(T["curve_cap"])

with col2:
    st.markdown(T["grid_title"])
    piv = grid.pivot(index="k", columns="t", values="fp").round(3)
    piv.columns = [f"t={c}" for c in piv.columns]
    st.dataframe(piv.style.format("{:.1%}", na_rep="—"))
    st.caption(T["grid_cap"])

st.divider()
st.caption(T["footer"])
