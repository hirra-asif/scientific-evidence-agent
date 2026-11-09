# Evaluation and reproduction

The [README](../README.md) summarises the main results. This document describes how the experiments were run, how outputs were scored and how to reproduce the recorded comparisons.

## Guideline comparison

Initial experiments compared the original Skill with no Skill on 30 development claims, repeated three times. The original Skill had lower evidence F1 in all three repetitions, while document/verdict results varied. A separate six-claim experiment with fixed abstracts suggested testing a shorter-evidence rule.

The revised rule was then compared with the original on **30 different development claims**, selected before the run: 10 SUPPORT, 10 CONTRADICT and 10 NOT_ENOUGH_INFO. Both conditions used the same model, tools and limits. The agent received only the claim text; dataset labels and annotated evidence stayed in the evaluator.

The following scores use the **29 claims that completed successfully in both conditions**:

| Metric | Original guidelines | Minimal-evidence guidelines |
|---|---:|---:|
| Document/verdict F1 | 0.488 | 0.600 |
| Abstract with evidence F1 | 0.439 | 0.550 |
| Evidence precision | 0.288 | 0.425 |
| Evidence recall | 0.548 | 0.548 |
| Evidence F1 | 0.378 | 0.479 |

Across all 30 attempts, exact matches for the complete set of documents and verdicts increased from **17/30 to 19/30**, counting failures as incorrect. Correct abstentions on NOT_ENOUGH_INFO claims were **8/10** in both conditions. Processing failures were **1/30** with the original guidelines and **0/30** with the revision.

Two additional claims achieved an exact document/verdict match: claim 56 changed from an abstention to the annotated SUPPORT verdict, and claim 218 completed successfully after the original-guideline run failed reference validation. This does not mean only two answers changed. Among the 29 successfully completed pairs, seven claims had different positive document/verdict sets; claim 218 was a separate processing failure in the original condition.

The revised rule was adopted after reviewing changed answers. This is a small development-set comparison, not a full SciFact benchmark result or a statistical claim of general improvement.

[Full comparison results](../results/confirmation_20261108_235629_639604.json) include outputs, tool traces, errors, timings, estimated costs and the frozen experiment settings. The saved condition named `current` refers to the original guidelines.

## Evidence-removal diagnostic

Three reviewed claim–abstract examples were tested under three conditions, each repeated three times:

| Condition | Expected behaviour | Results |
|---|---|---:|
| Full abstract | Original supported/refuted verdict | 9/9 |
| Findings removed | NOT_ENOUGH_INFO | 9/9 |
| Nonessential sentence removed | Original verdict preserved | 9/9 |

All **27 responses** matched the specified expectations, with no processing failures. Each request used a fresh context; expected answers, condition names and paper titles were withheld.

This tests the model's judgement of supplied text using the adopted guidelines. It does not test retrieval, and the repetitions are not independent claims.

[Full evidence-removal results](../results/evidence_removal_20261109_001511_824088.json).

## Examples from error analysis

These examples come from the [first original-Skill versus no-Skill comparison](../results/skill_comparison_20261108_224008_665417.json):

- **Extra citations:** On claim 137, the annotated evidence was sentences 7 and 9. The no-Skill run selected those two sentences; the Skill run selected 4, 6, 7, 8 and 9. The extra citations lowered precision and placed sentence 9 beyond the first three citations considered for abstract-with-evidence scoring.
- **Caution and annotation agreement:** On claim 72, the Skill run abstained, matching the dataset's NOT_ENOUGH_INFO label. On claim 692, it also abstained, while the dataset labelled the evidence CONTRADICT. These disagreements need inspection of the claim and study; abstention alone is not evidence of better judgement.
- **An alternative source:** On claim 148, the no-Skill run cited an unannotated rat-liver study reporting age-related decreases in autophagy-related proteins. It received no credit for that document. An alternative paper may warrant review, but relevance alone does not establish that it supports the full claim.
- **A coverage limit:** Claim 873 has five annotated abstracts, while the agent can read at most three. It cannot recover all five in one run. However, unchanged aggregate evidence recall between guideline versions does not establish that retrieval is the main bottleneck; verdict and citation errors also affect that score.

## Scoring

[scoring.py](../scoring.py) implements rules based on the [SciFact reference evaluator](https://github.com/allenai/scifact-evaluator). Document/verdict scoring checks the selected papers and labels. Sentence credit requires the correct label and a complete annotated evidence set. Precision measures credited predictions; recall measures recovered annotations; F1 balances the two. Counts are pooled before calculating these scores.

The 12 scoring tests cover alternative evidence sets, incomplete evidence, extra citations, wrong documents or labels, duplicate references and the reference evaluator's worked example.

For abstract-with-evidence scoring, only the first three citations are considered, unless the shortest annotated evidence set requires more than three sentences. Sentence-level scoring considers every citation.

Agreement with annotations is not the same as scientific correctness. In confirmation claim 56, a SUPPORT verdict matched the dataset while its explanation acknowledged an unproven causal link. Reference validation checks that citations exist and were read; it cannot establish that an explanation logically supports its verdict.

## Run controls and logging

The agent allows eight turns and three distinct abstracts per run. API retries are capped at two. Its 120-second timeout covers waiting for a response to begin; it is not a deadline for the whole agent run.

Validation rejects duplicate references, invalid sentence IDs, assessments of unread abstracts, and inconsistent verdict/evidence lists. The current agent comparison runners save tool activity and any returned output before validation, so rejected reports can be investigated. The earliest baseline runs predate this detailed logging.

## Reproduce the experiments

Follow the [setup instructions](../README.md#setup) first. Run commands from the repository root.

For the current evidence-removal test:

~~~bash
python -m experiments.evidence_test --repeats 3
~~~

This makes 27 API requests before any retries.

Historical comparisons require their original code and Skill versions. Their commands below use the file layout at those commits. Use a separate checkout and the same dependency/data setup:

| Experiment | Git revision | Command |
|---|---|---|
| Repeat the original baseline twice | `70eda83` | `python repeat_evaluation.py --baseline results/skill_comparison_20261108_224008_665417.json --repeats 2` |
| Fixed-abstract guideline comparison | `73909da` | `python compare_fixed_evidence.py --repeats 3` |
| Fresh-claim agent comparison | `3124ccf` | `python confirm_evidence_rules.py` |

These revisions match the source-file checksums saved with each experiment. Copy `requirements.txt` and `.env.example` from the latest revision when setting up these older checkouts. Keep the committed claim-selection files; the preparation scripts deliberately refuse to overwrite them. The historical comparison scripts intentionally reject a changed Skill, so running them on the latest revision is not a reproduction of those experiments.

The main comparison and evidence-removal results record package versions and source-file checksums. The confirmation and supplied-text experiments also save the exact Skill text or prompts. New runs may differ because model outputs are variable. Printed costs are SDK estimates, not billing records.

## Earlier demonstrations

`experiments/check_claim.py` and `experiments/evaluate.py` are early demonstrations kept for development history. The former checks one supplied abstract; the latter runs a small training-set check. The results reported above come from the later comparison and evidence-removal scripts.
