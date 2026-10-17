# Laya/JEV — Small TRAIN-only diagnostic

**Date:** 18 October 2026  
**Purpose:** Investigate whether missing evidence or missing decision rules explain failures in prior Laya choice corrections.  
**Disposition:** Exploratory diagnosis completed. **No new algorithm is justified yet.**

## Scope and safeguards

- Inspected the previously frozen, already-scored **H6 and H8 TRAIN tranches**, 100 distinct cases per tranche, containing **150 choice decisions each** (300 total). These are not a newly generated or independently held-out confirmation set.
- H6 source: `h6_fresh_validation_100_model_inputs.json` plus `h6_baseline_choice_predictions.json`; H8 source: `h7_fresh_validation_100_model_inputs.json` plus `h7_baseline_choice_predictions.json`. Both belong to the already archived experiment series.
- Used **only** `Laya-sol-phase1/benchmark/frozen/train_gold.jsonl` from the restricted ZIP to verify aggregate correctness. Did **not** read or score official TEST gold, official TEST predictions, H9, or new model outputs.
- No new LLM calls, parameter changes, code changes to repositories, or scoring submissions. Protected case-level correct answers were not included in this report.
- ZIP SHA-256: general `2c4b2f58098baf3a5ea71ab9114f7605cad3a5d3746b343fe2718a748699d0ba`; restricted `e67e8769778f9c1b79ca809daf596583915cf0c62fc331dba52a59f6baca58e8`.

## 1. Aggregate accuracy reproduced from archived TRAIN labels

| Archived experiment | Choice decisions correct | Accuracy |
|---|---:|---:|
| H6 baseline (not H6 candidate) | 101 / 150 | 67.3% |
| H8 baseline (not H8 routed candidate) | 105 / 150 | 70.0% |
| **Combined, descriptive only** | **206 / 300** | **68.7%** |

The combined rate is **not** the official M3 benchmark and must not be reported as an independent test score. These baselines are choice-specialist outputs, not full three-specialist routed M3 runs.

## 2. Errors are highly uneven across workflows

| Workflow / decision type | Correct / total | Accuracy |
|---|---:|---:|
| Agent trace — action | 27 / 50 | **54%** |
| Agent trace — outcome | 40 / 50 | 80% |
| Customer service — next action | 40 / 50 | 80% |
| Customer service — issue category | 46 / 50 | 92% |
| **Invoice processing — disposition** | **17 / 50** | **34%** |
| Security incidents — disposition | 36 / 50 | 72% |

**Strongest descriptive finding:** invoice disposition was much weaker than all other task categories in these two archived tranches. It is not a novel scientific effect, and other training splits could behave differently.

### Invoice disposition: what the model and teacher preferred

| Disposition | Synthetic teacher's hard labels | Laya baseline predictions |
|---|---:|---:|
| Approve | 18 | 5 |
| Manual review | 24 | 5 |
| Hold | 7 | 23 |
| Reject | 1 | 17 |

Laya chose **hold/reject in 40/50 invoices (80%)**, while the synthetic teacher chose those labels in **8/50 (16%)**. This is a sizable *directional mismatch*: Laya often selects more restrictive dispositions than the teacher labels. It does **not** prove that Laya's decisions were objectively wrong under a real finance policy. One plausible explanation is that the benchmark's implied approval/escalation thresholds are not fully specified in the input.

## 3. Missing-evidence micro-check (input-only annotation)

A deterministic SHA-256 ranking selected **16 choice questions** (two cases per workflow in each of H6/H8). One analyst classified each **from input states and question descriptions, without consulting its individual prediction or teacher label before assigning a category**. The annotator had previously viewed aggregate performance; this is *not an independently blinded study*.

| Judgment from input/question alone | Sampled decisions | Wrong against teacher |
|---|---:|---:|
| Missing a relevant factual observation (e.g., task completion or missing order confirmation) | 4 | 3 |
| Missing an explicit action/policy threshold (e.g., when to hold versus reject, investigate versus contain) | 8 | 3 |
| Sufficiently explicit for a defensible choice | 3 | 1 |
| Overlapping question category definitions | 1 | 0 |

**Interpretation:** there are identifiable inputs where needed information or escalation rules are unstated. However, this *very small, subjective and stratified* sample does not establish that missing information is the primary cause of Laya errors. In fact, several questions classified as missing a policy threshold still matched teacher labels. No statistical significance or population estimate is claimed.

A useful counterexample is agent-trace **outcome**: state records include aggregate step/error/violation counts but no explicit `task_completed` field; nevertheless, 40/50 outcome choices matched the synthetic teacher. So simple field absence is not sufficient to predict an error.

## 4. Is uncertainty informative?

Across the 300 archived choice decisions:

| Measure (descriptive mean) | Teacher-correct predictions (206) | Teacher-incorrect predictions (94) |
|---|---:|---:|
| Synthetic-teacher normalized distribution entropy | 0.560 | 0.667 |
| Laya's **self-reported** selected-label probability | 0.716 | 0.611 |

Teacher disagreement/uncertainty is greater on average for mistakes, and Laya's emitted probability is somewhat lower for mistakes. **Neither is a calibrated selection rule.** The self-reported numbers are model-generated response fields, not token probabilities, and workflow differences confound the aggregate association. No cutoff was tuned or tested.

## 5. Decision and next minimal step

**Result:** The *Missing-Evidence Boundary* is a plausible partial explanation, **not a confirmed dominant mechanism**. The more immediate, measurable finding is that invoice actions have a very large **teacher-versus-model policy/disposition mismatch**, while label uncertainty and decision underspecification may contribute.

**NO-GO for an algorithm or novelty claim.** If continuing, run a **small independent rubric audit**, not more corrective prompting:

1. Pick a fresh unused **TRAIN** set, reserve and freeze the IDs; focus on invoice disposition plus a smaller agent-action control.
2. Have **two independent reviewers**, blind to model choices and teacher labels, annotate whether an explicit factual or policy basis uniquely determines each response. Record disagreement and justifications.
3. Compare that adjudication with the synthetic-teacher labels and with Laya's frozen predictions **only after annotation is locked**. If the benchmark's label depends on unstated conventions, report a benchmark-specification gap; do not call it a new reasoning algorithm.
4. Do not touch official TEST gold or H9 for this diagnosis. Do not use these inspected TRAIN cases to tune a new scoring rule and then describe them as independent validation.

**Stop condition:** If reviewers can consistently and directly justify the teacher labels using the stated criteria and available input, the missing-evidence explanation is weakened. If reviewers cannot agree without introducing outside rules, the next investigation belongs to benchmark definition/label policy, not automatically to model invention.

### Reproduction

Run `python laya_train_diagnostic.py` beside the two uploaded ZIP archives. The script opens only the TRAIN gold file inside the restricted archive; it emits aggregate counts, not case-level protected labels. The 16 manually assigned classes are frozen inside the script so their recorded outcomes can be reproduced, but those classifications remain subjective.
