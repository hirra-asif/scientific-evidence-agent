# Scientific Evidence Agent

A Python agent that checks scientific claims against **5,183 research abstracts from SciFact**. Built with the Claude Agent SDK, it searches for relevant papers and returns a verdict for each abstract it reads, with evidence quoted directly from the source.

The project examines whether evidence guidelines improve claim checking, and whether answers change appropriately when supporting findings are removed.

## How it works

1. **Search:** Claude calls a BM25 keyword search over paper titles and abstracts.
2. **Read:** It selects up to three abstracts and receives their numbered sentences.
3. **Assess:** It returns `SUPPORT`, `CONTRADICT` or `NOT_ENOUGH_INFO` for each abstract.
4. **Validate:** Python checks the document references, sentence IDs and verdict/evidence consistency.
5. **Quote:** Evidence text is copied from the corpus using the validated sentence IDs.

The search and read functions are exposed through an **in-process MCP server**: a standard interface that lets the agent call Python tools. A [Skill](.claude/skills/scientific-evidence/SKILL.md) supplies reusable instructions about study populations, causation, numerical findings and evidence sufficiency.

The adopted rule asks for the smallest set of sentences that justifies the verdict, while retaining necessary qualifications. Verdicts apply to individual abstracts; the agent does not produce a consensus judgement across the literature.

## Setup

Tested on macOS with **Python 3.13.2**. The recorded experiments used `claude-haiku-5-5`. Running the agent requires an Anthropic API key with access to that model and incurs API charges.

From the repository root:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
~~~

Add your API key to the local `.env` file. Git ignores this file.

Download the dataset using the [SciFact maintainers' download](https://github.com/allenai/scifact#dataset). Skip this step if the data is already present:

~~~bash
curl -fL --retry 2 https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o scifact-data.tar.gz
tar -xzf scifact-data.tar.gz
rm scifact-data.tar.gz
~~~

The project expects `data/corpus.jsonl`, `data/claims_train.jsonl` and `data/claims_dev.jsonl`. The dataset is downloaded separately and is not committed.

Check the data and run the scoring tests without API calls:

~~~bash
python inspect_data.py
python -m unittest discover -s tests -v
~~~

## Run the agent

Run the built-in example:

~~~bash
python agent.py --skill on
~~~

The output includes search/read activity, a structured report, exact evidence quotations and an estimated API cost. To disable the guidelines, use `--skill off`.

To check another claim:

~~~python
import asyncio
from agent import main

report = asyncio.run(
    main(claim="CX3CR1 on the Th2 cells promotes airway inflammation.")
)
~~~

The agent allows eight turns and three distinct abstracts per run. API retries are capped at two. Its 120-second timeout covers waiting for a response to begin; it is not a deadline for the whole agent run.

Validation rejects duplicate references, invalid sentence IDs, assessments of unread abstracts, and inconsistent verdict/evidence lists. The current agent comparison runners save tool activity and any returned output before validation, so rejected reports can be investigated. The earliest baseline runs predate this detailed logging.

## Evaluation results

### Comparing evidence guidelines

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

The revised rule was adopted after reviewing changed answers. This is a small development-set comparison, not a full SciFact benchmark result or a statistical claim of general improvement.

[Full comparison results](results/confirmation_20261108_235629_639604.json) include outputs, tool traces, errors, timings, estimated costs and the frozen experiment settings. The saved condition named `current` refers to the original guidelines.

### Removing evidence

Three reviewed claim–abstract examples were tested under three conditions, each repeated three times:

| Condition | Expected behaviour | Results |
|---|---|---:|
| Full abstract | Original supported/refuted verdict | 9/9 |
| Findings removed | NOT_ENOUGH_INFO | 9/9 |
| Nonessential sentence removed | Original verdict preserved | 9/9 |

All **27 responses** matched the specified expectations, with no processing failures. Each request used a fresh context; expected answers, condition names and paper titles were withheld.

This tests the model's judgement of supplied text using the adopted guidelines. It does not test retrieval, and the repetitions are not independent claims.

[Full evidence-removal results](results/evidence_removal_20261109_001511_824088.json).

## Scoring and reproduction

[scoring.py](scoring.py) implements rules based on the [SciFact reference evaluator](https://github.com/allenai/scifact-evaluator). Document/verdict scoring checks the selected papers and labels. Sentence credit requires the correct label and a complete annotated evidence set. Precision measures credited predictions; recall measures recovered annotations; F1 balances the two. Counts are pooled before calculating these scores.

The 12 scoring tests cover alternative evidence sets, incomplete evidence, extra citations, wrong documents or labels, duplicate references and the reference evaluator's worked example.

Run experiment modules from the repository root. For the current evidence-removal test:

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

## Limitations

- Search can miss relevant papers, and the three-abstract limit restricts coverage. The corpus contains abstracts rather than full papers or a live literature search.
- Evaluation samples are small and selected from development data. Different claim IDs can still share papers or topics.
- Matching an annotation does not guarantee a scientifically sound interpretation. In one reviewed case, a SUPPORT verdict matched the dataset while its explanation acknowledged an unproven causal link.
- Reference validation checks that citations exist and were read. It cannot establish that an explanation logically supports its verdict.
- The evidence-removal results cover three selected examples. This is a research prototype, not a clinical decision tool.

## Main files

| File | Purpose |
|---|---|
| [agent.py](agent.py) | Agent loop, tool wrappers and output validation |
| [corpus_tools.py](corpus_tools.py) | Abstract search and reading |
| [Skill instructions](.claude/skills/scientific-evidence/SKILL.md) | Scientific evidence guidelines |
| [scoring.py](scoring.py) and [tests](tests/) | Evaluation rules and scoring tests |
| [experiments/](experiments/) | Evaluation runners and frozen claim-selection files |
| [confirm_evidence_rules.py](experiments/confirm_evidence_rules.py) | Original-versus-revised Skill comparison |
| [evidence_test.py](experiments/evidence_test.py) | Controlled evidence-removal checks |
| [results](results/) | Saved experimental outputs |

`experiments/check_claim.py` and `experiments/evaluate.py` are early demonstrations kept for development history. The former checks one supplied abstract; the latter runs a small training-set check. The results reported above come from the later comparison and evidence-removal scripts.

## References

- Wadden et al. (2020), [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).
- [SciFact data, documentation and terms](https://github.com/allenai/scifact).
- [SciFact reference evaluator](https://github.com/allenai/scifact-evaluator).
- [Claude Agent SDK documentation](https://code.claude.com/docs/en/agent-sdk/overview).
