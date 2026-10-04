#!/usr/bin/env python3
"""Analyze real Laya premature-closure benchmark output."""
import argparse, json, csv
from collections import defaultdict
from pathlib import Path

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def pct(a,b): return 0.0 if not b else 100*a/b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results/laya_results.jsonl")
    ap.add_argument("--csv", default="results/laya_summary.csv")
    args = ap.parse_args()
    rows = load(args.results)

    traj = defaultdict(list)
    for r in rows:
        key=(r["case_id"],r["order_name"],r["criteria_order"],r["repeat"])
        traj[key].append(r)
    for v in traj.values(): v.sort(key=lambda x:x["step"])

    finals=[v[-1] for v in traj.values()]
    by_case_repeat_criteria=defaultdict(dict)
    for r in finals:
        key=(r["case_id"],r["repeat"],r["criteria_order"])
        by_case_repeat_criteria[key][r["order_name"]]=r["choice"]

    order_sensitive=sum(1 for x in by_case_repeat_criteria.values() if len(set(x.values()))>1)

    # A/B presentation control: same case, evidence order and repeat; compare AB vs BA criteria.
    paired=defaultdict(dict)
    for r in finals:
        paired[(r["case_id"],r["order_name"],r["repeat"])][r["criteria_order"]]=r
    option_flip=0; option_pairs=0
    for d in paired.values():
        if "AB" in d and "BA" in d:
            option_pairs += 1
            option_flip += int(d["AB"]["choice"] != d["BA"]["choice"])

    # Repeat instability: deterministic systems should usually be stable; report instead of assuming.
    repeated=defaultdict(set)
    for r in finals:
        repeated[(r["case_id"],r["order_name"],r["criteria_order"])].add(r["choice"])
    repeat_unstable=sum(1 for s in repeated.values() if len(s)>1)

    # Revision: wrong immediately before first strong B evidence (weight >=4 in dataset convention)
    # Since weights aren't stored in results, E4/E5 are the strong items for v0.
    revision_opp=revision_success=sticky=0
    for t in traj.values():
        first_strong_idx=None
        for i,r in enumerate(t):
            if r["visible_evidence_ids"][-1] in {"E4","E5"}:
                first_strong_idx=i; break
        if first_strong_idx is not None and first_strong_idx>0:
            before=t[first_strong_idx-1]
            final=t[-1]
            if before["choice"] != before["gold_answer"]:
                revision_opp += 1
                if final["choice"] == final["gold_answer"]: revision_success += 1
                else: sticky += 1

    correct=sum(r["correct"] for r in finals)
    summary=[
        ("checkpoint_decisions",len(rows)),
        ("trajectories",len(traj)),
        ("final_accuracy_pct",round(pct(correct,len(finals)),2)),
        ("order_sensitive_case_repeat_criteria_pct",round(pct(order_sensitive,len(by_case_repeat_criteria)),2)),
        ("option_order_flip_pct",round(pct(option_flip,option_pairs),2)),
        ("repeat_unstable_pct",round(pct(repeat_unstable,len(repeated)),2)),
        ("revision_opportunities",revision_opp),
        ("revision_success_pct",round(pct(revision_success,revision_opp),2)),
        ("sticky_wrong_pct",round(pct(sticky,revision_opp),2)),
    ]

    print("=== Laya Premature-Closure v0 ===")
    for k,v in summary: print(f"{k}: {v}")

    print("\n=== Final accuracy by evidence order ===")
    by_order=defaultdict(list)
    for r in finals: by_order[r["order_name"]].append(r)
    for k in sorted(by_order):
        rs=by_order[k]; c=sum(r["correct"] for r in rs)
        print(f"{k:14s} {c}/{len(rs)} ({pct(c,len(rs)):.1f}%)")

    p=Path(args.csv); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["metric","value"]); w.writerows(summary)
    print(f"\nWrote {p}")

if __name__ == "__main__":
    main()
