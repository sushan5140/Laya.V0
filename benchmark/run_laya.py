#!/usr/bin/env python3
"""Run the premature-closure benchmark directly against Laya Router.predict()."""

import argparse, json, time
from pathlib import Path
from collections import OrderedDict

from laya import Router

ORDERS = {
    "original": [0,1,2,3,4],
    "reversed": [4,3,2,1,0],
    "interleaved": [0,3,1,4,2],
    "strong_first": [4,0,1,2,3],
    "strong_last": [1,2,0,3,4],
}

def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def render_state(case, visible):
    return {
        "question": case["question"],
        "evidence": [ev["text"] for ev in visible],
    }

def build_question(case, swap=False):
    if not swap:
        criteria = OrderedDict([
            ("A", case["option_a"]),
            ("B", case["option_b"]),
        ])
    else:
        criteria = OrderedDict([
            ("B", case["option_b"]),
            ("A", case["option_a"]),
        ])
    return {
        "decision": {
            "type": "choice",
            "instructions": (
                "Using only the evidence currently provided, which option is better supported? "
                "Do not assume evidence that is not shown."
            ),
            "criteria": criteria,
        }
    }

def run_one(router, case, order_name, order, swap, repeat, model=None):
    ordered = [case["evidence"][i] for i in order]
    rows = []
    for step in range(1, 6):
        visible = ordered[:step]
        state = render_state(case, visible)
        questions = build_question(case, swap=swap)
        t0 = time.perf_counter()
        kwargs = {"model": model} if model else {}
        result = router.predict(state, questions, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        ans = result["answers"]["decision"]
        choice = ans["choice"]
        probs = ans.get("probabilities", {})
        rows.append({
            "case_id": case["id"],
            "domain": case["domain"],
            "gold_answer": case["gold_answer"],
            "order_name": order_name,
            "order_evidence_ids": [e["id"] for e in ordered],
            "step": step,
            "visible_evidence_ids": [e["id"] for e in visible],
            "choice": choice,
            "p_A": probs.get("A"),
            "p_B": probs.get("B"),
            "confidence": ans.get("confidence"),
            "answer_confidence": ans.get("answer_confidence"),
            "correct": choice == case["gold_answer"],
            "criteria_order": "BA" if swap else "AB",
            "repeat": repeat,
            "routing_model": result.get("routing", {}).get("model"),
            "routing_repo": result.get("routing", {}).get("repo"),
            "usage": result.get("usage"),
            "elapsed_ms": round(elapsed_ms, 3),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v0.jsonl")
    ap.add_argument("--output", default="results/laya_results.jsonl")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default=None, help="Optional Laya model override, e.g. english or typed-decisions")
    ap.add_argument("--device", default=None, help="Optional device, e.g. cpu or cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--orders", nargs="*", default=list(ORDERS))
    args = ap.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[:args.limit]

    router_kwargs = {}
    if args.device:
        router_kwargs["device"] = args.device
    router = Router(**router_kwargs)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = len(cases) * len(args.orders) * 2 * args.repeats * 5
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for repeat in range(1, args.repeats + 1):
            for case in cases:
                for order_name in args.orders:
                    order = ORDERS[order_name]
                    for swap in (False, True):
                        for row in run_one(router, case, order_name, order, swap, repeat, args.model):
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            f.flush()
                            n += 1
                            print(
                                f"[{n}/{total}] {case['id']} r{repeat} {order_name} "
                                f"{row['criteria_order']} step={row['step']} -> "
                                f"{row['choice']} "
                                f"(P={row['answer_confidence']})"
                            )
    print(f"Saved {n} Laya checkpoint decisions to {out}")

if __name__ == "__main__":
    main()
