# When the Baseline Also Abstains: Pitfalls in Evaluating Abstention for Document Audit

**Authors:** Marcos de Paula (FLARE26 Project)
**Status:** Working draft (applied / evaluation-methodology paper). The
contribution is an empirical study and a set of evaluation lessons, with an
ontology-gated auditor as the case study.

---

## Abstract

Retrieval-augmented (RAG) auditors invent disagreements: asked for a value the
document never gives, they answer anyway, producing **false-positive
divergences**. We build an **ontology-gated** auditor that abstains on
type/scope mismatch — and, more importantly, we find that *measuring* whether
such a system abstains well is itself error-prone. Using a benchmark of
Brazilian contracts and government tender notices, we surface **three pitfalls**,
each illustrated by a concrete mistake we made and corrected: (i) a
**single-sample illusion**, where a rare false-positive rate reads as "0%" until
bootstrapped, and a self-consistency "free lunch" turns out to be sampling
noise; (ii) that **"hard" is defined by the model, not the taxonomy** — a naive
baseline ties an abstaining system not only on wholly-missing concepts but even
on taxonomically-adjacent ones (we expected a two-domain benchmark to
discriminate; the engineering half did not), so a gate helps *only* where the
model actually conflates a concept pair; and (iii) a **keyword-absence annotation
trap**, where labeling `ABSTAIN` from a missing keyword is invalid on real legal
text that paraphrases concepts. We
distill the lessons into a checklist for evaluating abstention in document
audit, and release the system, benchmark, and evaluation harnesses.

---

## 1. Introduction

Organizations compare many structured-but-unstructured documents — contracts,
tender notices (*editais*), financial statements — to find discrepancies. A
natural approach is RAG: retrieve relevant passages and ask an LLM to extract the
answer per document, then compare. This fails in a specific, costly way:
**standard RAG prefers answering to abstaining.** Asked "what is the penalty for
*total non-performance*?" of a contract that only sets a *late-payment* penalty,
a naive extractor returns the late-payment value — fabricating an answer to a
question the document never addresses. Across documents, such fabrications become
**false-positive divergences**: flagged disagreements that are artifacts of
hallucination, and in auditing the costliest error — worse than an honest "not
found, please verify."

The obvious fix is to make the system abstain. We do: Section 3 describes an
**ontology-gated** extractor that answers only when the evidence matches the
question's *type* and *scope*. But our main finding is methodological. **Whether
an abstaining system is actually better is surprisingly hard to measure**, and
the natural ways to measure it quietly mislead. We report three such pitfalls,
each one a mistake we made first and caught later:

- **Pitfall A — the single-sample illusion (§4.2).** Our first runs showed
  "0% ± 0%" false positives. Bootstrapping over more samples revealed the true
  rate is ~1–2%; the zero was sampling noise on a rare event. The same illusion
  made a permissive self-consistency vote look like free recall.
- **Pitfall B — "hard" is the model's, not the taxonomy's (§4.3).** A baseline
  ties our gate not only where a concept is wholly absent but even on
  taxonomically-adjacent pairs (a 21-case engineering benchmark we built to
  discriminate did not). A gate helps only where the *model* conflates a pair, so
  a benchmark must establish difficulty empirically, not by assuming adjacency.
- **Pitfall C — the keyword-absence annotation trap (§4.4).** We labeled
  `ABSTAIN` on real legal text by the absence of a keyword. Three of those labels
  were wrong: the concept was present under another name (*juros de mora* as
  *compensação financeira*; "payment term" as "payment within 30 days"). The gate
  had answered correctly; our benchmark was lying.

**Contributions.** (1) An ontology-gated auditor with explicit abstention and a
deterministic comparison judge (the case study, Section 3). (2) Three evaluation
pitfalls for abstention in document audit, each demonstrated on a real failure
(Section 4). (3) A checklist for evaluating abstention (Section 5), plus a
released benchmark and harnesses.

## 2. Related Work

**RAG and its hallucinations** [Lewis et al., 2020; survey: Gao et al., 2023].
Our concern is the failure mode of answering despite absent in-scope evidence.
**Structured extraction** (function-calling, Instructor [Instructor, 2023],
Guardrails [Guardrails AI, 2023], constrained decoding [Willard & Louf, 2023])
coerces typed outputs; we make the schema's *primary* job an abstention
pre-condition. **Selective prediction / abstention** [Geifman & El-Yaniv, 2017;
Kadavath et al., 2022] and answerability in reading comprehension [Rajpurkar et
al., 2018] are the closest neighbors; we study how to *evaluate* abstention in a
multi-document audit setting, not a new abstention method. **Faithfulness and RAG
evaluation** [Ji et al., 2023; Maynez et al., 2020; Es et al., 2024 (RAGAS)] ask
whether an answer is supported; we target the dual question — whether a system
correctly *declines* — and the ways benchmarks for it can mislead. **Legal NLP**
[Hendrycks et al., 2021 (CUAD); Chalkidis et al., 2020 (LEGAL-BERT)] supplies the
domain; our contribution is an abstention-centered evaluation in Brazilian
Portuguese procurement text.

## 3. The Case-Study System

FLARE26 is a RAG pipeline used here as a vehicle for the evaluation study:
hybrid retrieval (M1.5) → ontology-gated extraction (M2) → deterministic N-way
judge (M4) → summary (M5). Every verdict carries provenance to the source span.

**Ontology-gated extraction (M2).** The extractor emits a typed record with two
independent checks: **type/instrument** (the named instrument the question asks
about — e.g. a *penalty* vs. *interest*, distinct even when both are
percentages; must match) and **scope/condition** (the specific condition the
value applies to; strict equality required). If either fails, or a negative
restriction is violated, the record collapses to an explicit *evidence gap*. The
compatibility rule is **domain-agnostic** — stated over abstract instruments P/Q
and conditions A/B, with no hard-coded legal terms.

**Deterministic N-way judge (M4).** Given the typed answer per document, the
judge groups documents by equivalent answer (Brazilian number normalization:
`R$ 1.200.000,50` → `1200000.5`; `12,5%` ≡ `12.5`), verdict *consensus* /
*divergence* / *evidence gap* in O(N), no LLM call. Because it sees only
typed/abstained outputs, an upstream hallucination is the only way to fabricate a
divergence — which is what the gate suppresses.

**Optional self-consistency.** The extractor is run k times and votes; the vote
threshold t is a precision/recall knob (§4.2).

## 4. Evaluation and Three Pitfalls

**Benchmark.** 61 verified `(question, document)` pairs across two domains and two
languages: a **legal** split — 30 synthetic Brazilian contracts (13 `ABSTAIN`, 17
answerable) and 10 real tender notices (4 `ABSTAIN`, 6 answerable), in Portuguese
— and an **engineering** split — 21 English reliability/mechanical spec sheets
(12 `ABSTAIN`, 9 answerable), with both *type* near-neighbors (MTBF vs. MTTR,
tensile vs. yield, FMEA severity vs. occurrence) and *scope* near-neighbors (same
metric, different condition: tensile at 20°C vs. 200°C, torque for bolt class 8.8
vs. 10.9). Extractor: gpt-4o-mini, temperature 0, fixed seed. **Metrics:**
false-positive rate (fraction of `ABSTAIN` items answered — the headline),
abstention recall, answer recall.

### 4.1 The synthetic result (what we wanted to report)

| System | False-positive ↓ | Answer recall ↑ |
|--------|------------------|-----------------|
| Baseline (free extraction) | 38% ± 0% | 100% ± 0% |
| FLARE26 (gated)            | ~1–2%    | 81% ± 6% |

Ontological gating cuts false-positive divergences ~20–30× on the synthetic set.
An ablation confirms the strict scope rule matters: relaxing scope to accept
sub/superset conditions recovers recall but **reopens** false positives (0% →
38%), so the strict rule is the operating point.

### 4.2 Pitfall A — the single-sample illusion

Our first measurement reported **0% ± 0%** false positives, and a self-consistency
vote "answer if ≥1 of k runs answers" appeared to recover recall **for free**.
Both were artifacts of measuring a rare event once. Bootstrapping the per-item
sample pool exposes the truth:

| k (LLM calls/doc) | False-positive | Answer recall |
|-------------------|----------------|---------------|
| 1  | 1% ± 2% | 81% ± 6% |
| 5  | 3% ± 4% | 91% ± 3% |
| 10 | 5% ± 4% | 93% ± 2% |

The gate leaks rarely (~1–2% per sample), and the permissive "≥1" vote
*amplifies* it as k grows — recall and false positives rise together. Sweeping
the **vote threshold** t ("answer only if ≥ t of k") shows "≥1" was the wrong
knob [Wang et al., 2023]; a majority-style t filters the rare leaks:

| Policy | False-positive | Answer recall |
|--------|----------------|---------------|
| single call (k=1)      | 2%  | 77% |
| k=10, t=1 ("≥1")       | 10% | 86% |
| k=10, t=3              | 1%  | 82% |
| **k=10, t=4**          | **0%** | **80%** |

(The two tables are independent bootstrap runs; the small disagreement at k=1 —
1% vs. 2% false positives, 81% vs. 77% recall — is itself the sampling noise this
section is about.) The point k=10, t=4 dominates the single call on both axes.
**Lesson:** report rare-event metrics as mean ± std over repeated runs; sweep the
vote threshold; never trust a single "0%".

### 4.3 Pitfall B — "hard" is defined by the model, not the taxonomy

The synthetic legal benchmark's 38% baseline false-positive rate comes from
absences where a *similar-but-wrong concept is present* — asked for a *penalty*,
the document has *interest*, which the baseline grabs. It is tempting to call
such cases "hard" because the concepts are taxonomically adjacent. They are not
hard for that reason; they are hard because **the model conflates the pair**. Two
controls make this precise.

**Wholly-absent concepts are easy — even for the baseline.** Our first genuine
real-document `ABSTAIN` cases had the concept *wholly missing* (no indemnity
clause at all). Retrieval returns nothing relevant and **even the baseline
abstains**, tying the gate at 0% false positives. A 10-pair real pilot (4 such
absences, 6 answerable) gives FLARE 0% / 83% recall vs. baseline 0% / 50%: the
gate wins on recall, but the false-positive comparison is uninformative.

**Taxonomically-adjacent ≠ hard.** We built 21 engineering cases with both *type*
near-neighbors (MTBF vs. MTTR, tensile vs. yield, severity vs. occurrence) and
*scope* near-neighbors (tensile at 20°C vs. 200°C; torque for class 8.8 vs.
10.9). We expected the baseline to leak. **It did not** — on all 12 engineering
`ABSTAIN` cases the baseline correctly abstained (0% false positives), matching
the gate. The model simply does not confuse these crisp technical concepts. A
short control isolates the cause: the legal *interest-for-penalty* leak persists
when the same case is translated to English (the baseline answers "interest of
1% per month" for a *penalty* question), so it is the **concept pair**, not the
language or the legal domain, that the model treats as interchangeable.

**Consequences.** (i) An abstention gate provides a measurable advantage *only
where the baseline leaks*, which depends on the model's concept-confusability —
not on the annotator's taxonomy, the domain, or the language. (ii) In crisp
domains the baseline is already careful, so the gate's benefit is small or nil
there — the engineering split shows the gate does no harm (0% false positives,
100% recall, no over-abstention) but adds nothing. (iii) Showing a gate's value
therefore requires first *finding* the concept pairs a given model conflates,
which may be rare. **Lesson:** report results per concept-difficulty, established
empirically (does the baseline leak?), not by assuming adjacency implies
difficulty.

### 4.4 Pitfall C — the keyword-absence annotation trap

We initially labeled real-document `ABSTAIN` from the absence of a *keyword*.
Three labels were wrong, because real legal text paraphrases concepts: *juros de
mora* (interest) appeared as *compensação financeira*; "payment term" as
"payment will be made within 30 days". In each case the gate had **answered
correctly** and our label called it a false positive. Re-annotating absence by
**concept** — exhausting synonyms/paraphrases before declaring `ABSTAIN` —
corrected them; one genuinely ambiguous case (*compensação financeira* ≈
interest?) is held out pending domain adjudication. **Lesson:** never infer
absence from a missing keyword on real text; annotate by concept, and use a
second annotator.

### 4.5 Separating retrieval from the gate

Diagnosing a real-document recall failure revealed a **retrieval** bug, not a
gate failure: the context assembler truncated an *unordered set*, occasionally
dropping the highest-ranked lexical block (the exact penalty clause). A
relevance-ordered merge raised real-document recall from 67% to 83% at unchanged
false positives. **Lesson:** in error analysis, attribute failures to retrieval
vs. extraction before blaming the gate.

## 5. Recommendations: a checklist for evaluating abstention

1. **Repeat and bootstrap.** Report false positives as mean ± std over k runs;
   a single "0%" on a rare event is an illusion (§4.2).
2. **Include hard absences.** Cover cases where a similar-but-wrong concept is
   present, not only wholly-absent concepts; report easy vs. hard separately
   (§4.3).
3. **Annotate by concept, not keyword.** Confirm absence by exhausting
   paraphrases; prefer a second annotator on real text (§4.4).
4. **Sweep the vote threshold.** For self-consistency, "answer if any" inflates
   false positives; report the (k, t) operating curve (§4.2).
5. **Attribute failures.** Separate retrieval misses from gate decisions before
   drawing conclusions (§4.5).

## 6. Discussion and Limitations

- **Small, partly synthetic benchmark.** A ~1–2% false-positive rate cannot be
  estimated tightly from 13–17 `ABSTAIN` items; absolute rates are indicative.
- **Real-document external validity is open.** The real `ABSTAIN` cases are easy;
  the hard real cases coincide with the ambiguous ones awaiting domain
  adjudication. This is the priority extension.
- **Single annotator; proprietary LLM.** No inter-annotator agreement yet;
  open-model replication is future work.
- The system is an **architectural pattern**, not a new ML method; the paper's
  weight is on the evaluation lessons and the released artifacts.

## 7. Conclusion

We set out to show that an ontology-gated auditor avoids false-positive
divergences, and it does cut them ~20–30× on a synthetic benchmark. But the more
durable contribution is a caution: evaluating abstention in document audit is
easy to get wrong. A rare false-positive rate looks like zero until you
bootstrap; a naive baseline ties an abstaining system on easy absences; and
labeling absence by keyword silently corrupts a benchmark on real legal text. We
hope the checklist, benchmark, and harnesses help others measure abstention
honestly. Future work: hard real-document absences, a second annotator, and
open-model replication.

## References

> **Note:** real references to the authors' best knowledge; verify all
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
