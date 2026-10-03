# Laya.V0 — Premature Closure Benchmark

This repository currently contains a small research harness for testing the hypothesis:

> Does an AI system commit to an answer too early, then underweight stronger contradictory evidence that arrives later?

## What is included

- `benchmark/dataset_v0.jsonl` — 20 controlled synthetic cases.
- `benchmark/run_experiment.py` — feeds evidence incrementally and records answer/confidence after every prefix.
- The runner repeats each case with different evidence orders.
- Results are saved as JSONL.
- `benchmark/analyze_results.py` — compares final answers, revision behavior, and order sensitivity.

## Experimental idea

For each case the model sees evidence one item at a time:

```
E1
E1 + E2
E1 + E2 + E3
E1 + E2 + E3 + E4
E1 + E2 + E3 + E4 + E5
```

The same evidence is then reordered and evaluated again.

If the final answer or willingness to revise changes systematically with evidence order, that is evidence of order sensitivity / premature closure.

## Important

The 20 cases are deliberately synthetic. They are for validating the experimental protocol, not for making real-world factual claims.

The original Laya source has not yet been pushed into this GitHub repository; GitHub reported the repo as empty when this benchmark was added. The benchmark is therefore isolated under `benchmark/` so it can later be connected to the actual Laya inference interface without rewriting the experiment.

## Quick start

```bash
cd benchmark
pip install -r requirements.txt

# Validate the whole pipeline without an API
python run_experiment.py --provider mock

# Run against an OpenAI-compatible model configured through the OpenAI SDK
export OPENAI_API_KEY=...
python run_experiment.py --provider openai --model gpt-5.6

# Analyze
python analyze_results.py results/results.jsonl
```

The mock provider is only a pipeline check. It is not a research result.
