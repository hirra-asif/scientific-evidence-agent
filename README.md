# Scientific Evidence Agent

An agent that checks scientific claims against **5,183 research abstracts from SciFact**. Built with the Claude Agent SDK, it searches for relevant papers, reads up to three abstracts and returns a verdict for each, with evidence quoted directly from the source.

**Results:** On a 30-claim development sample, 29 claims completed successfully with both guideline versions. The minimal-evidence rule increased evidence precision from **0.288 to 0.425**, with recall unchanged at **0.548**. One run failed with the original guidelines; none failed with the revision. In a separate supplied-text test, the model returned `NOT_ENOUGH_INFO` after findings were removed in all nine runs: three examples, each repeated three times. These are small development experiments, not a full SciFact benchmark evaluation. [Full results and methods](docs/evaluation.md).

## Why I built this

I’m interested in how language models can help people work with scientific evidence. I built this project to learn the Claude Agent SDK and test whether evidence guidelines make an agent’s answers more reliable. I wanted to check whether it could find the right evidence and recognise when the available text was insufficient. I developed it step by step, adding search tools, reference checks and evaluations.

## How it works

1. **Search:** Claude calls a BM25 keyword search over paper titles and abstracts.
2. **Read:** It selects up to three abstracts and receives their numbered sentences.
3. **Assess:** It returns `SUPPORT`, `CONTRADICT` or `NOT_ENOUGH_INFO` for each abstract.
4. **Validate:** Python checks the document references, sentence IDs and verdict/evidence consistency.
5. **Quote:** Evidence text is copied from the corpus using the validated sentence IDs.

The tools run through an **in-process MCP server**, which lets the agent call Python functions. A [Skill](.claude/skills/scientific-evidence/SKILL.md) supplies reusable instructions about study populations, causation, numerical findings and evidence sufficiency.

Verdicts apply to individual abstracts; the agent does not produce a consensus judgement across the literature.

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

## Limitations

- Keyword search can miss relevant papers, and the three-abstract limit restricts coverage. The corpus contains abstracts, not full papers or a live literature search.
- Evaluation samples are small and drawn from development data. Different claim IDs can still share papers or topics.
- Agreement with dataset annotations does not guarantee a scientifically sound interpretation. Valid citations also do not establish that an explanation supports its verdict.
- The evidence-removal diagnostic covers three selected examples. This is a research prototype, not a clinical decision tool.

## Main files

| File | Purpose |
|---|---|
| [agent.py](agent.py) | Agent loop, tool wrappers and output validation |
| [corpus_tools.py](corpus_tools.py) | Abstract search and reading |
| [Skill instructions](.claude/skills/scientific-evidence/SKILL.md) | Scientific evidence guidelines |
| [scoring.py](scoring.py) and [tests](tests/) | Evaluation rules and scoring tests |
| [experiments/](experiments/) | Evaluation runners and frozen claim-selection files |
| [results](results/) | Saved experimental outputs |

## References

- Wadden et al. (2020), [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).
- [SciFact data, documentation and terms](https://github.com/allenai/scifact).
- [SciFact reference evaluator](https://github.com/allenai/scifact-evaluator).
- [Claude Agent SDK documentation](https://code.claude.com/docs/en/agent-sdk/overview).
