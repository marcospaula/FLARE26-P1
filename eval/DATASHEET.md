# Datasheet — FLARE26 Abstention Benchmark

A small benchmark for **ontological abstention** in document audit: given a
question and a document, the system must return the correct answer **or**
abstain (`ABSTAIN`) when the document does not contain the requested information.
File: [`gold_dataset.json`](gold_dataset.json).

## Motivation

Multi-document audit systems built on RAG tend to hallucinate an answer when the
requested datum is absent, producing **false-positive divergences**. This
benchmark measures whether a system abstains correctly, with a focus on
*discriminating* cases where a similar-but-wrong concept is present.

## Composition

- **40 verified** `(question, document)` pairs (+1 pending domain review).
  - **30 synthetic** pairs over small, hand-built contracts (`documentos/Contrato_*`,
    `Edital_PGFN_*`). 13 `ABSTAIN`, 17 answerable.
  - **10 real** pairs over two Brazilian government tender notices
    (`pe-srrf01-02_2020...`, `edital-04-2025...`). 4 `ABSTAIN`, 6 answerable.
- Each item has: `id`, `pergunta` (question), `documento` (PDF filename),
  `gold` (answer string or `"ABSTAIN"`), `evidencia` (supporting/justifying
  text), `verificado` (bool).
- Domains/concepts covered: penalties (multa), interest (juros de mora), payment
  terms, warranty terms, retention, indemnity, LGPD/data-protection, contract
  duration. Language: **Brazilian Portuguese**.
- Discriminating distinctions by design: *interest ≠ penalty*, *warranty ≠
  payment*, *total non-performance ≠ late payment*, *monetary correction ≠
  interest*.

## Collection & annotation process

- Synthetic documents were authored to isolate specific concepts; their labels
  follow directly from the text.
- Real documents are public Brazilian procurement notices. Their labels were
  produced by reading the source and, for `ABSTAIN`, **confirming concept
  absence by exhausting synonyms/paraphrases** — not by keyword absence.
- **Lesson encoded in the data.** Three initial real-doc `ABSTAIN` labels were
  wrong because absence was inferred from a missing keyword while the concept
  was present under another name (e.g. *juros de mora* → *compensação
  financeira*; "payment term" → "payment within 30 days"). These were corrected;
  one ambiguous case (`real-srrf-juros`) is held out (`verificado=false`) pending
  domain adjudication.

## Recommended uses & metrics

- **False-positive rate** (headline): fraction of `ABSTAIN` items the system
  nonetheless answers.
- **Abstention recall** and **answer recall**.
- Harnesses in this folder report mean ± std with bootstrap (`run_eval_agg.py`),
  a (k,t) self-consistency sweep (`run_eval_grid.py`), and a recall×cost curve
  (`run_eval_curve.py`).

## Limitations & cautions

- **Small.** A ~1–2% false-positive rate cannot be estimated tightly from 13–17
  `ABSTAIN` items; treat absolute rates as indicative.
- **Partly synthetic.** The synthetic split is clean by construction; the real
  split is a *pilot*. The real `ABSTAIN` cases are "easy" (concept wholly
  absent) — the *hard* real cases (similar-but-wrong concept present) are not yet
  covered and are the priority extension.
- **Single annotator.** No inter-annotator agreement yet; a second annotator is
  planned.
- Labels reflect a careful but non-lawyer reading; legal nuances (e.g. interest
  vs. financial compensation) may warrant expert review before high-stakes use.

## Maintenance

Versioned with the repository. Extensions (hard real cases, second annotator,
more domains) are tracked in the project notes.
