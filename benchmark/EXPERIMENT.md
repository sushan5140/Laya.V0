# Experiment v0 Design

## Hypothesis

A model can become committed to an early answer after several weak pieces of evidence and then underweight stronger contradictory evidence that arrives later.

## Dataset

20 synthetic binary-decision cases. Every case contains exactly five evidence items.

- Three weak items support A.
- Two stronger items support B.
- The intended answer after all evidence is B.
- Strength metadata exists only for the mock pipeline validator. Real model prompts do not expose the numeric weights.

## Five evidence orders

1. `original` — weak A evidence first, strong B evidence later.
2. `reversed` — exact reverse.
3. `interleaved` — weak and strong evidence mixed.
4. `strong_first` — strongest B evidence first.
5. `strong_last` — strongest B evidence last after the weak A cluster.

## Checkpoints

For each order, evaluate after each prefix:

- 1 evidence item
- 2 evidence items
- 3 evidence items
- 4 evidence items
- all 5 evidence items

20 cases × 5 orders × 5 checkpoints = **500 model decisions** per model.

## Recorded fields

Each checkpoint stores:

- answer (A/B)
- confidence (0–1)
- correctness
- evidence IDs and order
- which evidence arrived most recently
- provider/model identifier
- raw model response

## Primary v0 signals

### Order sensitivity
Does the final answer change when the exact same evidence is reordered?

### Revision success
If the model is wrong before strong contradictory evidence arrives, does it eventually switch to the gold answer?

### Sticky-wrong behavior
Does it remain wrong even after seeing all stronger contradictory evidence?

### Premature confidence
Was it already highly confident (>= 0.75 in the initial descriptive analysis) in the wrong answer before the strong contradiction arrived?

The 0.75 threshold is exploratory for v0, not a preregistered scientific cutoff.

## Next integration step

Once the actual Laya source is present in this repository, add a provider adapter that maps the same `predict(case, evidence)` interface onto Laya's inference API. The dataset and analysis code should not need to change.
