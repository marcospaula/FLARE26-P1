# FLARE26 — Ontology-Gated Abstention for Document Audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20881699.svg)](https://doi.org/10.5281/zenodo.20881699)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A neuro-symbolic "glass-box" auditor that compares documents and knows when *not* to answer.**

FLARE26 extracts a typed answer from each document, **abstains** when the
requested information is genuinely absent or of the wrong *type/scope*, and then
decides consensus vs. divergence across documents with a **deterministic** judge.
The goal is to avoid the most expensive error in multi-document audit: a
**false-positive divergence** — a disagreement the system *invents* by
hallucinating an answer that the document never gave.

> **Honest framing.** This is an applied / systems project plus an empirical
> study, not a new ML method. Its value is in the architectural pattern
> (*abstention as a pre-condition for comparison*) and in an evaluation that is
> reported transparently, including its negative results and limitations.

---

## Key result (synthetic benchmark, 30 pairs)

Extractor: `gpt-4o-mini`, temperature 0, fixed seed. Mean ± std.

| System | False-positive ↓ | Answer recall ↑ |
|--------|------------------|-----------------|
| Baseline (free extraction) | 38% ± 0% | 100% ± 0% |
| FLARE26 (ontology-gated)   | **~1–2%** | 81% ± 6% |
| FLARE26 + self-consistency (k=10, t=4) | **~0%** | 80% |

Ontological gating cuts the false-positive divergence rate **~20–30×** vs. the
baseline. Self-consistency exposes a tunable recall × precision operating curve
(a **majority-style vote** keeps false positives near zero). See
[`paper/draft.md`](paper/draft.md) §5.

**A real-document pilot** (two Brazilian government tender notices) is preliminary
and reported with caveats in §6 — chiefly that *reliable absence annotation* on
real legal text (keyword-absence ≠ concept-absence) is the binding constraint.

---

## How it works

| Module | Role |
|--------|------|
| **M1.5** | Hybrid retrieval (dense vectors + lexical), parent/child chunking |
| **M2**   | **Ontology-gated extraction** — answers only if the evidence matches the question's *type* **and** *scope*; otherwise emits an explicit *evidence gap* |
| **M4**   | **Deterministic N-way judge** — groups documents by equivalent answer (BR number normalization), verdict in O(N), no LLM call |
| **M5**   | Executive summary, anchored to the judge's typed output |

The compatibility rule is **domain-agnostic** (stated over abstract instruments
P/Q and conditions A/B — no hard-coded legal terms).

## Repository layout

```
flare26_core.py        # Pure, tested logic (number normalization, judges) — no Streamlit/OpenAI
flare26_pipeline.py    # Headless RAG (M1.5/M2) — callable without a UI
app_flare26_pdf.py     # Streamlit "glass-box" app (UI shell over the pipeline)
tests/                 # pytest suite (42 tests; regression over a ledger snapshot)
eval/                  # Benchmark + reproducible evaluation harnesses (see eval/README.md)
  gold_dataset.json    #   61 verified (question, document) pairs, two domains/two languages
  run_eval*.py         #   metrics, k-repetition, (k,t) vote sweep, recall×cost curve
paper/                 # Paper draft (Markdown + LaTeX), references, build script
documentos/            # Sample contracts and tender notices (PDF)
```

## Quick start

Requires Python 3.11+ (developed on 3.14) and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# --- run the tests (no API key needed) ---
python -m pytest tests/ -q

# --- run the evaluation (needs OPENAI_API_KEY) ---
export OPENAI_API_KEY=sk-...
python eval/run_eval.py                  # baseline vs FLARE26 on the gold set
FLARE_CONSENSO=5 python eval/run_eval.py # with self-consistency

# --- run the interactive app ---
echo 'OPENAI_API_KEY = "sk-..."' > .streamlit/secrets.toml
streamlit run app_flare26_pdf.py
```

## Benchmark

The evaluation benchmark ([`eval/gold_dataset.json`](eval/gold_dataset.json)) has
**61 verified** `(question, document)` pairs across **two domains and two
languages** — Brazilian legal contracts and tender notices (Portuguese: 30
synthetic + 10 real) and English reliability/mechanical engineering specs (21) —
each labeled with the correct answer **or** `ABSTAIN`. It deliberately includes
*discriminating* cases — e.g. *interest ≠ penalty*, *warranty term ≠ payment
term*, *MTBF ≠ MTTR*, *tensile ≠ yield* — where a naive extractor may
hallucinate. See [`eval/README.md`](eval/README.md) for the annotation protocol.

## Status & limitations (read this)

- The strong result is on a **small, partly synthetic** benchmark (30 pairs).
  Error bars on a ~1–2% false-positive rate are wide.
- The gate **leaks rarely** (it is not a 0% guarantee); the defensible claim is
  the large reduction vs. baseline.
- **Real-document external validity is not established.** The binding constraint
  is domain-grade *absence* annotation; this is the priority next step.
- Depends on a proprietary LLM (`gpt-4o-mini`); open-model replication is future
  work.

## Paper

The preprint **"When the Baseline Also Abstains: Pitfalls in Evaluating
Abstention for Document Audit"** is published on Zenodo (CC BY 4.0):
**[doi:10.5281/zenodo.20881699](https://doi.org/10.5281/zenodo.20881699)**.
Source (Markdown + LaTeX) and the compiled PDF are in [`paper/`](paper/).
Regenerate the reading PDF with:

```bash
python paper/md_to_html.py paper/draft.md /tmp/draft.html
soffice --headless --convert-to pdf --outdir paper /tmp/draft.html && mv paper/draft.pdf paper/
```

## Citation

If you use FLARE26 or its benchmark, please cite the preprint:

```bibtex
@misc{depaula2026flare26,
  author       = {de Paula, Marcos},
  title        = {When the Baseline Also Abstains: Pitfalls in Evaluating
                  Abstention for Document Audit},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20881699},
  url          = {https://doi.org/10.5281/zenodo.20881699}
}
```

## License

[MIT](LICENSE) © 2026 Marcos de Paula. The sample documents in `documentos/`
are synthetic or public Brazilian procurement notices, included for
reproducibility.
