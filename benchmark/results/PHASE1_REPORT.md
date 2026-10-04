# Laya Phase 1: evidence order and premature closure

Run on 2026-10-04 from `main` at `5431604b4c9a6321f05f92d576ac017ccb1f7a2c`. The benchmark implementation and dataset were not changed. The full real Laya output is [`laya_results.jsonl`](laya_results.jsonl); its SHA-256 is `fb818182130a6e55bcf83c0d5eea793104c128d0a9a5ad5b6f63ad6daf9d8b49` (dataset SHA-256 `509331c0ed862eea629d10d7ed25ea5dbf8058873d95ed0318793618c4958b76`).

## Execution

- Python 3.12.14, Linux x86_64, CPU with nine visible logical processors and no CUDA; `laya==0.3.26`, `torch==2.5.1+cpu`, `transformers==5.18.0`, `huggingface-hub==1.33.0`, `safetensors==0.8.0`, `numpy==2.5.3`.
- Router selected `english` at `convaiinnovations/laya` on every call; downloaded model snapshot `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- Set up with `python -m venv .venv-clean`, `.venv-clean/bin/python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu`, then `.venv-clean/bin/python -m pip install -r benchmark/requirements-laya.txt`. The first default `torch==2.14.1` environment caused a bus error on `import torch`; the clean CPU environment worked. **No source compatibility edits** were needed.
- From `benchmark/`, ran `../.venv-clean/bin/python -u run_laya.py --limit 2 --repeats 1` (100/100 smoke decisions), then `../.venv-clean/bin/python -u run_laya.py --repeats 3`, then `../.venv-clean/bin/python analyze_laya.py results/laya_results.jsonl`. After workspace recovery, the smoke test was repeated in a fresh `.venv` with `--output results/laya_smoke_results.jsonl` to preserve its log; all 100 choices and p(B) values match the corresponding full-run calls.
- **3,000/3,000 successful decisions**, zero failed or skipped: 20 cases × 5 orders × 2 criteria orders × 3 repeats × 5 checkpoints. Independent integrity audit found 3,000 unique complete keys; all choices were A/B, probabilities and both confidence fields were finite and in [0,1], `p_A+p_B≈1`, `answer_confidence` equaled the chosen probability, and no state tokens were dropped or inputs truncated.
- Laya warned that a checkpoint temperature bucket (`choice:11+`) was outside its accepted range and clamped from 0.1005828 to 0.5. Confidence for affected entries is uncalibrated; probability magnitudes are descriptive. `confidence` is Laya's normalized entropy score, while `answer_confidence` is mass on the chosen option.

## Results

| Measure | Result | Unit |
| --- | ---: | --- |
| Final accuracy | **429/600 = 71.5%** | 143/200 unique case × order × criteria outcomes, each repeated identically three times |
| Order sensitive | **24/60 = 40.0%** | Case × repeat × criteria groups with differing final choices across five orders; 11/20 distinct cases sensitive in at least one criteria order |
| All evidence-order pair flips | **82/400 = 20.5%** | Ten pairs × 20 cases × 2 criteria orders; pairs are dependent |
| AB/BA final-choice flips | **39/300 = 13.0%** | 13/100 unique case × order pairs |
| Repeat instability | **0/200** | Case × order × criteria groups changing final choice over three repeats |
| Revision opportunities / success | **333 / 216 = 64.86%** | 111 / 72 unique opportunities immediately before first E4/E5 |
| Sticky wrong | **117/333 = 35.14%** | 39/111 unique opportunities remained A after all evidence |

| Evidence order | Correct finals / 120 | Accuracy | Mean final p(B) |
| --- | ---: | ---: | ---: |
| original | 87 | 72.5% | .5856 |
| reversed | 90 | 75.0% | .6066 |
| strong_first | 96 | 80.0% | .6360 |
| strong_last | 96 | 80.0% | .5758 |
| interleaved | 60 | 50.0% | .4482 |

Interleaved is 22.5 points below original. Original versus interleaved flips 9/40 unique case × criteria final choices, all toward A. The 20.5% matched evidence-order pair disagreement is descriptively above the 13.0% AB/BA control. But strong-first and strong-last both reach 80%, so there is no consistent late-evidence penalty. BA puts gold B first: it changes 11/100 unique matched final choices from A to B and 2/100 from B to A. The 40% five-order group rate and 13% criteria pair rate have different denominators and should not be directly compared.

## Representative trajectories

Choices at steps 1–5 and p(B) at the same steps; all are AB criteria, repeat 1. At step 5 each order includes all five evidence items.

| Case | Order | Choices | p(B) |
| --- | --- | --- | --- |
| `pc_009` | original | AAABB | .094, .106, .154, .764, .631 |
| `pc_009` | interleaved | ABABA | .094, .553, .411, .606, .366 |
| `pc_003` | original | AAAAB | .354, .332, .384, .283, .771 |
| `pc_003` | reversed | BBBBB | .884, .532, .618, .525, .600 |
| `pc_001` | original | AAAAA | .027, .024, .026, .055, .108 |
| `pc_001` | strong_first | AAAAB | .441, .074, .038, .134, .511 |

## Judgment and next experiment

**Phase 1 weakly supports an evidence-order effect, but does not establish premature closure.** Each checkpoint calls `Router.predict()` afresh on the visible list; earlier answers and confidence are not passed forward. A final-order flip may be input-position sensitivity, not persistence of a prior belief. The revision and sticky-wrong counts likewise compare independent calls.

The 20 hand-written cases all have gold B and the same three weak A/two strong B structure. AB/BA reorders criteria but does not balance the semantic labels; original and strong-last both leave E4/E5 last. No independent human evidence-strength adjudication was performed. Repeats were deterministic, so the independent case count is 20, not 600. Calibration warning limits probability interpretation.

For Phase 2, pair a **stateful** arm that receives its prior choice/confidence/history at each step with a **stateless** arm receiving only accumulated evidence. Hold the final evidence multiset fixed, counterbalance semantic A/B labels and gold answers, vary decisive-evidence timing separately from weak-item order, include neutral/contradictory cases with checked strength, and preregister whether late evidence causes a larger final error or revision deficit in the stateful arm. Keep AB/BA and probability comparisons as controls.

Artifacts: [`laya_results.jsonl`](laya_results.jsonl), [`laya_summary.csv`](laya_summary.csv), [`laya_analysis.log`](laya_analysis.log), [`integrity_audit.log`](integrity_audit.log), [`environment.txt`](environment.txt), [`laya_smoke_results.jsonl`](laya_smoke_results.jsonl), and [`smoke.log`](smoke.log). A cloud workspace reset after the full run erased its original stdout log; the full JSONL had already been uploaded as a GitHub blob and was restored byte-for-byte, verified by SHA-256. The analyzer and integrity logs were regenerated from that exact output, and the smoke test was rerun. The lost full-run stdout log is not represented as preserved.
