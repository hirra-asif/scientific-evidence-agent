# Frozen dev evaluation

60 previously unused dev claims, 20 per class; two repetitions per condition (480 attempts).
Counts are pooled before calculating precision, recall and F1. Failed attempts retain their gold evidence, contribute no positive predictions and earn no correct-abstention credit.

## Performance

| Condition | Document/verdict P | R | F1 | Evidence P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_haiku | 0.622 | 0.554 | 0.586 | 0.490 | 0.428 | 0.457 |
| improved_haiku | 0.541 | 0.576 | 0.558 | 0.391 | 0.400 | 0.396 |
| improved_sonnet | 0.667 | 0.739 | 0.701 | 0.479 | 0.506 | 0.492 |
| fixed_pipeline | 0.671 | 0.620 | 0.644 | 0.474 | 0.406 | 0.437 |

## Reliability and resource use

| Condition | NEI false flags | Correct NEI abstentions | NEI failures | All failures | Mean searches | Mean abstracts | Mean seconds | Reported USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_haiku | 5/40 | 35/40 | 0 | 0/120 | 2.17 | 2.10 | 8.84 | 0.1333 |
| improved_haiku | 8/40 | 31/40 | 1 | 1/120 | 2.21 | 2.07 | 9.57 | 0.1371 |
| improved_sonnet | 7/40 | 33/40 | 0 | 0/120 | 1.19 | 1.62 | 8.01 | 1.5852 |
| fixed_pipeline | 7/40 | 33/40 | 0 | 0/120 | 1.00 | 3.00 | 7.95 | 0.1801 |

A false flag is any SUPPORT or CONTRADICT assessment on a gold NEI claim. Failed NEI attempts are counted separately. Latency includes retrieval and the SDK call; shared retriever setup is separate.

Attempts without SDK cost estimates: current_haiku: 0, improved_haiku: 0, improved_sonnet: 0, fixed_pipeline: 0.

## Planned paired comparisons

Differences are candidate minus reference. Both repetitions and all four conditions stay together when resampling a claim. The bootstrap resamples within each gold class (50,000 draws; seed 47).

| Comparison | Metric | Difference | 95% interval | Adjusted interval |
| --- | --- | ---: | --- | --- |
| improved_haiku − current_haiku | evidence_f1 | -0.061 | [-0.102, -0.023] | [-0.117, -0.011] |
| improved_haiku − current_haiku | document_verdict_f1 | -0.028 | [-0.075, 0.015] | [-0.093, 0.029] |
| fixed_pipeline − improved_haiku | evidence_f1 | +0.042 | [-0.033, 0.112] | [-0.058, 0.138] |
| fixed_pipeline − improved_haiku | document_verdict_f1 | +0.086 | [-0.027, 0.200] | [-0.065, 0.241] |
| improved_sonnet − improved_haiku | evidence_f1 | +0.096 | [0.025, 0.167] | [0.001, 0.192] |
| improved_sonnet − improved_haiku | document_verdict_f1 | +0.143 | [0.032, 0.259] | [-0.004, 0.300] |

The adjusted intervals use 99.167% confidence per endpoint (Bonferroni across three comparisons and two primary metrics). The frozen rule requires both adjusted intervals above zero to claim higher F1 on both primary metrics. Both below zero favors the reference; other outcomes are inconclusive or mixed.

- **improvements:** inconclusive or mixed.
- **pipeline_vs_agent:** inconclusive or mixed.
- **sonnet_vs_haiku:** inconclusive or mixed.

These conclusions concern the primary F1 metrics. NEI outcomes, failures and resource use remain descriptive and must also inform any deployment choice.

## Limits and reproducibility

- Balanced subset of 60 unused dev claims, not a full SciFact benchmark result.
- Unused refers to recorded project history; papers/topics can overlap development data.
- Bootstrap clusters repetitions by claim; dependence between different claims about the same paper is not modeled.
- Intervals are approximate; an interval containing zero does not establish equivalence.

- Result SHA-256: `33b6fce85a5034f369d67363fd64fd155e83799885969d5cdfe5a8faedef4ff9`
- Frozen manifest SHA-256: `1d125bc99931f66b2580772d365a953832791467f1b34bd71dd2cd0ad42ef42a`

[Frozen manifest](../experiments/final_evaluation.json) · [Raw results](../results/final_evaluation.json) · [Analysis, per-repeat scores and matched-success diagnostics](../results/final_evaluation_analysis.json)
