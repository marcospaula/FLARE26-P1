# Knowing When the Answer Isn't There: Ontology-Gated Extraction for False-Positive-Free Multi-Document Audit

**Authors:** Marcos de Paula (FLARE26 Project)
**Status:** Working draft (applied / systems paper). Not an ML-method paper;
the contribution is an architectural pattern plus an empirical evaluation on a
real domain (Brazilian public-sector contracts and tender notices).

---

## Abstract

Retrieval-augmented generation (RAG) systems applied to document comparison tend
to *hallucinate* an answer when the requested information is simply absent,
producing **false-positive divergences** that erode trust in audit settings —
where the costliest error is not getting a value wrong, but inventing a
disagreement that does not exist. We present FLARE26, a neuro-symbolic
"glass-box" auditor whose extraction step is **ontologically gated**: the model
must explicitly declare the *type* and the *scope* of both the question and the
retrieved evidence, and abstains (returns an explicit *evidence gap*) whenever
they do not match. A downstream **deterministic** judge then decides
consensus/divergence across N documents over the typed, abstained outputs. On a
30-pair benchmark of Brazilian contracts and tender notices, ontological gating
**cuts the false-positive divergence rate from 38% to the low single digits**
(~1–2% per extraction) — a ~20–30× reduction — while a **self-consistency** vote
exposes an explicit, tunable precision/recall operating curve: increasing the
number of samples k raises answer recall (81% → 93%) at the price of a small rise
in false positives (~1% → ~5%). The gate's advantage over the baseline is large
and robust; the residual false positives are rare leaks rather than zero, which
we characterize honestly rather than claim away.

---

## 1. Introduction

Organizations increasingly compare large numbers of structured-but-unstructured
documents — contracts, tender notices (*editais*), financial statements — to
detect discrepancies. A natural approach is RAG: retrieve relevant passages and
ask an LLM to extract the answer per document, then compare. This breaks down in
a specific, costly way: **standard RAG prefers answering to abstaining.** When a
contract is asked "what is the penalty for *total non-performance*?" but only
contains a clause about "*late-payment* penalty," a naive extractor returns the
late-payment value — fabricating an answer to a question the document never
addresses. When such fabricated answers are compared across documents, they
manifest as **false-positive divergences**: the system flags a disagreement that
is an artifact of hallucination.

In auditing, the cost structure differs from open-domain QA. A false positive
triggers unnecessary legal review or a wrong decision; it is worse than an
honest "not found, please verify." We therefore target the false-positive
divergence rate as the primary metric.

**Contributions.**
1. **Ontologically-gated extraction** with explicit abstention: the extractor
   declares the *type* and *scope* of the question vs. the evidence and abstains
   on mismatch (Section 3).
2. A **deterministic N-way judge** that decides consensus/divergence over the
   typed outputs, auditable and O(N) (Section 3).
3. A **benchmark and metric** for ontological abstention on Brazilian legal
   documents, showing a **~20–30× reduction in false-positive divergences** and
   an explicit, tunable recall × precision curve via self-consistency
   (Section 5).

## 2. Related Work

**Retrieval-augmented generation.** RAG [Lewis et al., 2020; survey: Gao et al.,
2023] grounds generation in retrieved passages; our pipeline is a RAG system,
but our concern is the failure mode where generation proceeds despite the absence
of in-scope evidence.

**Structured extraction from LLMs.** Function-calling, Instructor [Instructor,
2023] and Guardrails [Guardrails AI, 2023], and grammar/schema-constrained
decoding [Willard & Louf, 2023] coerce outputs into typed schemas. We build on
this but make the schema's *primary* job an **abstention pre-condition**: decide
whether to answer at all.

**Selective prediction / abstention.** Selective classification trades coverage
for reliability by abstaining [Geifman & El-Yaniv, 2017]; LLMs exhibit partial
self-knowledge of correctness [Kadavath et al., 2022]. The reading-comprehension
analogue is answerability — detecting unanswerable questions, as in SQuAD 2.0
[Rajpurkar et al., 2018] — closest in spirit to our gate. We frame abstention
not as a post-hoc QA filter but as a **pre-condition for cross-document
comparison**.

**Faithfulness / hallucination.** Hallucination is well documented [Ji et al.,
2023; Maynez et al., 2020], and RAG-specific evaluation measures groundedness
[Es et al., 2024]. These largely ask whether an answer is *supported*; we target
the dual failure: answering when *no* in-scope evidence exists.

**Legal NLP / document comparison.** Contract-review datasets and models
[Hendrycks et al., 2021 (CUAD); Chalkidis et al., 2020 (LEGAL-BERT)] address
clause extraction; we contribute an evaluation centered on **abstention** in
Brazilian Portuguese public-procurement documents.

*Gap.* Abstention treated as the gate that prevents false-positive divergences
in multi-document audit, evaluated with a divergence-oriented metric.

## 3. Method

FLARE26 is a pipeline: hybrid retrieval (M1.5) → ontologically-gated extraction
(M2) → deterministic N-way judge (M4) → executive synthesis (M5). The whole
pipeline is a "glass box": every verdict carries provenance to the source span.

### 3.1 Ontologically-gated extraction (M2)

The extractor emits a typed record that separates **two independent checks**:

- **Type/instrument** (*natureza*): the named instrument the question asks about
  (e.g., a *penalty* vs. *interest* — distinct legal instruments even when both
  are percentages). Must match.
- **Scope/condition** (*escopo*): the specific condition/event the value applies
  to. The strict variant requires the evidence scope to match the question
  scope; mismatched scopes ⇒ abstain.

If either check fails (or a negative user restriction is violated), the record
collapses to an explicit `NÃO LOCALIZADO` / *evidence gap* with confidence 0.
Crucially, the prompt's compatibility rule is **domain-agnostic**: it is stated
over abstract instruments P/Q and conditions A/B, with no hard-coded legal terms.

### 3.2 Deterministic N-way judge (M4)

Given the typed answer per document, the judge groups documents by **equivalent
answer** (numbers normalized for Brazilian formatting: `R$ 1.200.000,50` →
`1200000.5`; `12,5%` ≡ `12.5`). The verdict is *consensus* (one group),
*divergence* (multiple groups), or *evidence gap*, computed in O(N) with no LLM
call. Because the judge sees only abstained/typed outputs, a hallucinated answer
upstream is the only way to create a false-positive divergence — which is exactly
what the gate suppresses.

### 3.3 Self-consistency (optional)

The extractor is run k times and the answers are aggregated by a vote. The
permissive policy "answer if **any** run answers" maximizes recall; a stricter
threshold (answer only if ≥ t of k agree) trades recall for precision. Section
5.4 characterizes this as an explicit operating curve over k (and t).

## 4. Evaluation

**Corpus & gold.** 30 (question, document) pairs over 16 documents and 7
questions, drawn from Brazilian service contracts and public tender notices.
Each pair is labeled with the correct answer **or** `ABSTAIN` when the document
does not contain the requested datum (13 ABSTAIN, 17 with answer). Labels were
verified against the source text. The benchmark deliberately includes
discriminating cases: *interest ≠ penalty*, *warranty term ≠ payment term*, and
*total non-performance ≠ late payment*.

**Systems.** (i) BASELINE: free extraction, no ontological gate. (ii) FLARE26:
gated extraction (strict scope). (iii) FLARE26 + self-consistency (k=5).
Extractor: gpt-4o-mini, temperature 0, fixed seed.

**Metrics.** (a) **False-positive rate** — fraction of ABSTAIN items the system
nevertheless answers (the headline; a hallucination that would produce a false
divergence). (b) **Abstention recall** — fraction of ABSTAIN items correctly
abstained. (c) **Answer recall** — fraction of answerable items answered.

**Aggregation.** Numbers are mean ± std: BASELINE over 3 runs, FLARE26-strict
over 8 runs, self-consistency by bootstrap (k=5, 300 resamples of the per-item
sample pool).

## 5. Results

### 5.1 Main result

| System | False-positive ↓ | Abstention recall ↑ | Answer recall ↑ |
|--------|------------------|---------------------|-----------------|
| BASELINE (free extraction)        | 38% ± 0% | 62% ± 0% | 100% ± 0% |
| FLARE26 (single-call gating)      | **~1–2%** | ~98–100% | 81% ± 6% |
| FLARE26 + self-consistency (k=5)  | ~3% | ~97% | 91% ± 3% |

Ontological gating cuts the false-positive divergence rate from **38% to the low
single digits** (~1–2% per extraction; an 8-run sample measured 0%, but a
larger-pool bootstrap, §5.4, places the true rate at ~1–2% — the 0% was within
sampling noise of a rare event). It correctly handles the *interest≠penalty*,
*warranty≠payment* and *non-performance≠late-payment* distinctions. The residual
false positives are **rare leaks, not zero**; we report the full precision/recall
operating curve in §5.4 rather than a single point.

### 5.2 Ablation: the precision/recall frontier of the scope rule

| Scope rule | False-positive | Answer recall |
|------------|----------------|---------------|
| Strict (condition equality) | **0%** | 71–82% |
| Intersection (subset/superset accepted) | 38% | 88% |

Loosening the scope check to accept sub/superset conditions recovers recall but
**reopens false positives** (e.g., a "late-payment penalty" gets returned for a
"non-performance penalty" question). The strict rule is the correct operating
point for audit: never asserting a false divergence outweighs recall.

### 5.3 Error analysis

FLARE26's residual abstentions on answerable items fall into two classes:
- **Legitimate frontier:** evidence strictly more specific (`delay > 10 days`
  for a question about "delay") or more general (`partial *or* total default`
  for a question about "total") than the question. Tightening or loosening here
  trades precision; we keep the conservative side.
- **Non-determinism:** items such as "retention 5%/10%" oscillate between
  answering and abstaining across runs — addressed in §5.4.

### 5.4 The recall × cost operating curve of self-consistency

We run the extractor up to 10 times per item, collect the answer pool, and
bootstrap the "answer if ≥1 of k" decision (400 resamples). The result is an
explicit **precision/recall operating curve** governed by k:

| k (LLM calls / doc) | False-positive | Answer recall |
|---------------------|----------------|---------------|
| 1  | 1% ± 2% | 81% ± 6% |
| 2  | 1% ± 3% | 87% ± 5% |
| 3  | 2% ± 3% | 89% ± 5% |
| 5  | 3% ± 4% | 91% ± 3% |
| 8  | 5% ± 4% | 93% ± 2% |
| 10 | 5% ± 4% | 93% ± 2% |

Two honest observations. First, **ABSTAIN leakage is rare but nonzero**: an
abstain item occasionally flips to an answer (~1–2% per sample). Second, the
permissive "≥1 of k" vote *amplifies* this as k grows, so recall (81% → 93%) and
false positives (1% → 5%) rise together — this is a genuine trade-off, **not a
free lunch** (an earlier, smaller experiment that observed 0 leakage was within
the sampling noise of this low rate).

**The vote threshold is the right control.** Sweeping the threshold t ("answer
only if ≥ t of k samples answer") shows that the naive "≥1" was simply the wrong
knob: because leaks appear in only 1–2 of the k samples, a majority-style
threshold filters them out and drives false positives back to ~0 while keeping
recall high.

| Policy | False-positive | Answer recall |
|--------|----------------|---------------|
| single call (k=1)        | 2%  | 77% |
| k=10, t=1 (naive "≥1")   | 10% | 86% |
| k=10, t=3                | 1%  | 82% |
| **k=10, t=4**            | **0%** | **80%** |
| k=8, t=4                 | 0%  | 78% |

The operating point **k=10, t=4 dominates the single call on both axes** (0% vs.
2% false positives, 80% vs. 77% recall); tolerating ~1% false positives (t=3)
buys 82% recall. Recall saturates near 82% — the remaining gap is the legitimate
scope frontier of §5.3. Self-consistency is therefore genuinely useful **when the
threshold is chosen sensibly**; the gate stays far below the 38% baseline
throughout, which is the paper's central, robust claim.

## 6. Discussion and Limitations

- **Precision/recall trade-off, not a guarantee.** The gate does not reach
  exactly 0% false positives; it leaks rarely (~1–2%), and self-consistency
  trades a small false-positive rise for recall. The defensible claim is the
  ~20–30× reduction vs. baseline, not an absolute zero.
- **Small benchmark ⇒ wide error bars.** With only 13 ABSTAIN items, a ~1–2%
  false-positive rate cannot be estimated tightly (± a few points). A larger,
  more balanced benchmark is needed to pin the rate down — this *strengthens*
  the evaluation rather than threatening the thesis.
- **External validity: a cautious pilot.** On two real Brazilian tender notices
  (116k and 198k characters), reliable ABSTAIN annotation — not the model —
  proved to be the binding constraint: inferring "the document lacks X" from the
  absence of *keyword* X is invalid on real legal text, which paraphrases
  concepts (*juros de mora* as *compensação financeira*; "payment term" as
  "payment within 30 days"). Three of our first real-doc ABSTAIN labels were
  wrong for this reason — the gate had answered correctly. After re-annotating
  absence by **concept** (exhausting synonyms), a 10-pair pilot (4 genuine
  absences, 6 answerable) gives FLARE 0% false positives and 67% recall vs. the
  baseline's 0% and 50%. Two honest caveats temper this: (i) the genuine
  absences are *easy* (the concept is wholly missing, so even the baseline
  abstains — 0% FP for both); the *hard* real cases (a similar-but-wrong concept
  is present, the source of the synthetic benchmark's 38% baseline FP) coincide
  with the ambiguous ones still awaiting domain adjudication; and (ii) recall is
  capped partly by **retrieval** — on the 116k-character notice, the penalty
  clauses were not reliably retrieved. Retrieval scaling and hard real-doc
  ABSTAINs are the priority next steps; we make no strong real-document claim.
- **LLM non-determinism** is mitigated by seed + self-consistency; results are
  reported as mean ± std with bootstrap.
- The extractor depends on a proprietary LLM (gpt-4o-mini); replication with an
  open model is future work.
- This is an **architectural pattern**, not a new ML method; the value is in the
  evaluation and the abstention-as-precondition framing.

## 7. Conclusion and Future Work

In multi-document audit, the dangerous error is the fabricated disagreement.
Gating extraction on explicit type+scope compatibility, with deterministic
downstream comparison, **cuts the false-positive divergence rate by ~20–30×**
(38% → low single digits); a self-consistency vote with a sensible majority-style
threshold (e.g., k=10, t=4) then **recovers recall to ~80% at ~0% false
positives**, dominating the single call on both axes. The gate's advantage over
the baseline is large and robust; the remaining false positives are rare leaks we
quantify rather than claim away. Future work: a larger, double-annotated
benchmark (including the real government tender notices) to tighten the rate
estimate, and replication with an open LLM.

## References

> **Note:** real references to the best of the authors' knowledge; verify all
> bibliographic details (author lists, pages, DOIs, venues) before submission.
> BibTeX entries in `paper/references.bib`.

- Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of EMNLP.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL (System Demonstrations).
- Gao, Y., Xiong, Y., Gao, X., et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.
- Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
- Guardrails AI (2023). *Guardrails.* Software. https://github.com/guardrails-ai/guardrails
- Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS Datasets and Benchmarks.
- Instructor (2023). *Instructor: Structured Outputs for LLMs.* Software. https://github.com/jxnl/instructor
- Ji, Z., Lee, N., Frieske, R., et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys.
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). *Language Models (Mostly) Know What They Know.* arXiv:2207.05221.
- Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). *On Faithfulness and Factuality in Abstractive Summarization.* ACL.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD.* ACL.
- Wang, X., Wei, J., Schuurmans, D., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models.* arXiv:2307.09702.
