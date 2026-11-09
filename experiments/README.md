# Experiment index

Start with the [final results and methods](../docs/final_evaluation.md). The
default application is [agent.py](../agent.py); [app.py](../app.py) is its
interactive interface. This directory contains the evaluation history.

## Final four-condition comparison

| File | Purpose |
| --- | --- |
| [final_evaluation.json](final_evaluation.json) | Frozen claims, configurations, schedule, source hashes and decision rules |
| [prepare_final_evaluation.py](prepare_final_evaluation.py) | Audit previously used claims and create the manifest |
| [run_final_evaluation.py](run_final_evaluation.py) | Execute or resume all 480 scheduled attempts |
| [analyze_final_evaluation.py](analyze_final_evaluation.py) | Validate results and compute the predefined paired intervals |
| [final_evaluation_common.py](final_evaluation_common.py) | Claim selection, provenance checks and bootstrap implementation |
| [final_conditions.py](final_conditions.py) | Four configurations and the fixed pipeline's SDK assessment |
| [run_final_pilot.py](run_final_pilot.py) | Shared execution loop used by the pilot and final runner |

[Raw final results](../results/final_evaluation.json) ·
[Analysis JSON](../results/final_evaluation_analysis.json)

## Development history

| Stage | Entry points | Saved evidence |
| --- | --- | --- |
| Offline retrieval | `prepare_retrieval_split.py`, `prepare_retrieval_config.py`, `evaluate_bm25.py`, `evaluate_hybrid.py`, `validate_hybrid.py` | [Fixed split](retrieval_split.json), [E5 configuration](retrieval_config.json), [validation result](../results/retrieval_hybrid_validation.json) |
| Bounded agent development | `prepare_agent_training.py`, `compare_agent_retrieval.py`, `compare_claim_text_first.py` | [Training claims](agent_training_claims.json), [hybrid comparison](../results/agent_retrieval_training.json), [claim-text-first comparison](../results/agent_claim_text_first_training.json) |
| Technical pilot | `prepare_final_pilot.py`, `run_final_pilot.py` | [Pilot manifest](final_conditions_pilot.json), [24-attempt result](../results/final_conditions_pilot.json) |
| Earlier guideline experiments | `compare_skills.py`, `compare_fixed_evidence.py`, `confirm_evidence_rules.py`, `evidence_test.py` | [Methods and findings](../docs/evaluation.md) |
| Earlier sampling and diagnostics | `prepare_evaluation.py`, `prepare_confirmation.py`, `evaluate.py`, `repeat_evaluation.py`, `check_claim.py` | [Initial dev selection](evaluation_claims.json), [confirmation selection](confirmation_claims.json) |

The final runner still imports shared helpers from earlier modules, including
`agent_comparison.py` and `compare_agent_retrieval.py`. Its manifest also
verifies training, retrieval-validation and pilot results. These files remain
at their original paths; moving them into an archive would break those checks.
Historical results are retained alongside the final evaluation.

## Reproducing the frozen evaluation

The final experiment was frozen at commit
[`56e41db77a3faf713f9513654e21523f31e09ef8`](https://github.com/hirra-asif/scientific-evidence-agent/commit/56e41db77a3faf713f9513654e21523f31e09ef8).
The later lazy-loading refactor and UI are not part of that evaluated source
snapshot. Source checks intentionally reject running the old manifest against
modified code. Do not update the old manifest's fingerprints to bypass this.

Use a separate checkout of the frozen commit:

```bash
git worktree add --detach ../scientific-evidence-agent-frozen 56e41db77a3faf713f9513654e21523f31e09ef8
cd ../scientific-evidence-agent-frozen
```

The recorded environment was **Python 3.13.2 on Apple Silicon macOS**, with
the embedding device set to `mps`. Create a matching virtual environment,
install the original requirements and the exact package versions recorded in
`experiments/final_evaluation.json`, download the corpus, and configure your
local API key as described in that checkout's README. Corpus files and keys
are not committed or copied automatically between worktrees.

Then run:

```bash
python -m experiments.run_final_evaluation
python -m experiments.analyze_final_evaluation
```

This makes new paid model requests. The original observed outputs remain in
the main branch's `results/` directory. A rerun is a replication and is not
expected to reproduce stochastic model outputs exactly.

## Unit tests and demo

On the current branch, run `python -m unittest discover -s tests -v` without
corpus data or an API key. Tests use small fixtures and mocked SDK responses;
the recorded examples in the app come from real saved evaluation runs.

The [demo example bundle](../docs/demo_examples.json) contains exact excerpts
from `results/final_evaluation.json`, including abstracts captured by the fixed
pipeline. Tests verify its source hash, run records and document text. It is
display-only and is never used by live retrieval or evaluation.
