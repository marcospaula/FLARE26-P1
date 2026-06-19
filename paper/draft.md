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
**eliminates false-positive divergences (38% → 0%)** at the cost of answer
recall, and a **self-consistency** vote recovers recall to **92% ± 3% while
preserving 0% false positives**. We show this is possible because ontological
abstentions are *stable* (zero leakage across runs) while over-abstentions are
mere sampling noise.

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
   documents, and the finding that **self-consistency recovers recall at zero
   precision cost** because ontological abstentions are stable (Section 5).

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

The extractor is run k times; the system answers if **any** run answers,
abstaining only if all k abstain, and returns the most frequent answer value.
Section 5.4 shows why this asymmetric policy is safe.

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
| BASELINE (free extraction)        | 38% ± 0% | 62% ± 0%  | 100% ± 0% |
| FLARE26 (strict gating)           | **0% ± 0%** | **100% ± 0%** | 82% ± 4% |
| FLARE26 + self-consistency (k=5)  | **0% ± 0%** | **100% ± 0%** | **92% ± 3%** |

Ontological gating eliminates false-positive divergences (38% → 0%, ± 0%),
correctly handling the *interest≠penalty*, *warranty≠payment* and
*non-performance≠late-payment* distinctions. Self-consistency then recovers
answer recall to 92% ± 3% while keeping 0% false positives.

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

### 5.4 Self-consistency recovers recall at zero precision cost

The key diagnostic (5 runs per item): **all 13 ABSTAIN items showed zero
leakage** — they abstained in every run. The variance is entirely on the
answerable side (legitimate answers that sometimes flip to abstention). Because
ontological abstentions never leak, the most permissive vote — **answer if ≥1 of
k runs answers** — recovers recall without reopening false positives:

| System | False-positive | Answer recall |
|--------|----------------|---------------|
| FLARE26 (single sample, 8 runs)        | 0% ± 0% | 82% ± 4% |
| FLARE26 + self-consistency (k=5, ≥1)   | **0% ± 0%** | **92% ± 3%** |

The 0% false-positive guarantee is exact and stable (0% ± 0% across all runs)
because no ontological abstention leaks; larger k recovers more low-base-rate
borderline answers at linear LLM cost.

## 6. Discussion and Limitations

- **Recall < 100%** is the price of the 0%-FP guarantee; self-consistency raises
  it to 92% ± 3%, with residual loss on the legitimate sub/superset frontier.
- **LLM non-determinism** is mitigated by seed + self-consistency; results are
  reported as mean ± std.
- The corpus is partly synthetic; generalization to other domains and to the
  large real tender notices (present in the corpus but not yet labeled) remains
  to be validated, ideally with a second annotator.
- The extractor depends on a proprietary LLM (gpt-4o-mini); replication with an
  open model is future work.
- This is an **architectural pattern**, not a new ML method; the value is in the
  evaluation and the abstention-as-precondition framing.

## 7. Conclusion and Future Work

In multi-document audit, the dangerous error is the fabricated disagreement.
Gating extraction on explicit type+scope compatibility, with deterministic
downstream comparison, **eliminates false-positive divergences**; self-consistency
then **recovers recall at no precision cost**, because ontological abstentions are
stable while over-abstentions are noise. Future work: expand and double-annotate
the corpus (including the real government tender notices), sweep k for the
recall × cost curve, and replicate with an open LLM.

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
