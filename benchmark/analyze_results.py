#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="results/results.jsonl")
    args = parser.parse_args()
    rows = load_rows(args.results)

    trajectories = defaultdict(list)
    for row in rows:
        trajectories[(row["case_id"], row["order_name"])].append(row)
    for key in trajectories:
        trajectories[key].sort(key=lambda x: x["step"])

    final_rows = [traj[-1] for traj in trajectories.values()]
    final_correct = sum(r["correct"] for r in final_rows)

    by_case_final = defaultdict(dict)
    for r in final_rows:
        by_case_final[r["case_id"]][r["order_name"]] = r["answer"]

    order_sensitive_cases = 0
    for case_id, answers in by_case_final.items():
        if len(set(answers.values())) > 1:
            order_sensitive_cases += 1

    revision_opportunities = 0
    successful_revisions = 0
    sticky_wrong = 0
    high_conf_wrong_before_strong = 0

    for traj in trajectories.values():
        gold = traj[0]["gold_answer"]
        first_gold_step = None
        for r in traj:
            if r["last_evidence_stance"] == gold and r["last_evidence_weight"] >= 4:
                first_gold_step = r["step"]
                break

        if first_gold_step is not None and first_gold_step > 1:
            before = traj[first_gold_step - 2]
            after = traj[-1]
            if before["answer"] != gold:
                revision_opportunities += 1
                if after["answer"] == gold:
                    successful_revisions += 1
                else:
                    sticky_wrong += 1
                if before["confidence"] >= 0.75:
                    high_conf_wrong_before_strong += 1

    print("=== Premature Closure Benchmark Summary ===")
    print(f"Checkpoint predictions: {len(rows)}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Cases: {len(by_case_final)}")
    print(f"Final accuracy: {final_correct}/{len(final_rows)} ({pct(final_correct, len(final_rows)):.1f}%)")
    print(f"Order-sensitive cases: {order_sensitive_cases}/{len(by_case_final)} ({pct(order_sensitive_cases, len(by_case_final)):.1f}%)")
    print(f"Revision opportunities: {revision_opportunities}")
    print(f"Successful revisions: {successful_revisions}/{revision_opportunities} ({pct(successful_revisions, revision_opportunities):.1f}%)")
    print(f"Sticky-wrong trajectories: {sticky_wrong}/{revision_opportunities} ({pct(sticky_wrong, revision_opportunities):.1f}%)")
    print(f"High-confidence wrong beliefs before strong contradiction: {high_conf_wrong_before_strong}")

    print("\n=== Final accuracy by order ===")
    by_order = defaultdict(list)
    for r in final_rows:
        by_order[r["order_name"]].append(r)
    for order_name in sorted(by_order):
        rs = by_order[order_name]
        correct = sum(r["correct"] for r in rs)
        print(f"{order_name:14s} {correct}/{len(rs)} ({pct(correct, len(rs)):.1f}%)")

    print("\nInterpretation:")
    print("- 'Order-sensitive cases' means identical evidence produced different final A/B answers under different orders.")
    print("- 'Sticky-wrong' means the system was wrong before strong contradictory evidence arrived and still ended wrong.")
    print("- The mock provider is only a harness check; do not treat mock numbers as scientific findings.")

if __name__ == "__main__":
    main()
