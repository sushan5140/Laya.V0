# Real Laya Phase-1 Run

This is the real Laya adapter for the premature-closure hypothesis.

## Why this adapter uses Laya's native probability

For `choice` answers Laya returns:

- `choice`
- `probabilities`
- `confidence`
- `answer_confidence`

Use `answer_confidence` as the probability mass on the reported answer. Laya's `confidence` for choice tasks is normalized entropy, so it is not interchangeable with `answer_confidence`.

## Controls built in

### 1. Evidence-order manipulation
Each case is tested as:

- original
- reversed
- interleaved
- strong_first
- strong_last

### 2. Option-position control
Every trajectory is run twice:

- criteria ordered A then B
- criteria ordered B then A

If the result changes only because the option labels were reordered, that is presentation bias, not evidence-order premature closure.

### 3. Repeats
Default: 3 repeats.

This lets us report instability instead of silently assuming deterministic behavior.

## Scale

Default full run:

20 cases × 5 evidence orders × 2 criteria orders × 3 repeats × 5 checkpoints

= **3,000 Laya decisions**.

A fast pilot:

```bash
python run_laya.py --limit 2 --repeats 1
```

Full run:

```bash
python run_laya.py --repeats 3
python analyze_laya.py results/laya_results.jsonl
```

CPU explicitly:

```bash
python run_laya.py --device cpu --repeats 3
```

GPU:

```bash
python run_laya.py --device cuda --repeats 3
```

## Scientific interpretation

A strong Phase-1 result requires all three:

1. Same evidence gives materially different final answers/probabilities under different evidence orders.
2. The effect is larger than the A/B criteria-position control.
3. The effect reproduces across cases/repeats rather than depending on one example.

Do **not** call mock-provider results evidence. Only real Laya runs count.
