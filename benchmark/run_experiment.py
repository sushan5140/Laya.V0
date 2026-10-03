#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from pathlib import Path

ORDERS = {
    "original": [0, 1, 2, 3, 4],
    "reversed": [4, 3, 2, 1, 0],
    "interleaved": [0, 3, 1, 4, 2],
    "strong_first": [4, 0, 1, 2, 3],
    "strong_last": [1, 2, 0, 3, 4],
}

SYSTEM_PROMPT = """You are taking part in a controlled evidence-integration experiment.
Use ONLY the evidence currently shown. Do not assume unseen evidence.
Return strict JSON with exactly two keys:
{"answer":"A" or "B","confidence": number from 0 to 1}
Confidence means your confidence that the chosen option is better supported by the evidence currently visible.
Do not add prose outside JSON."""

def load_cases(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases

def build_prompt(case, evidence):
    lines = [
        f"Question: {case['question']}",
        f"Option A: {case['option_a']}",
        f"Option B: {case['option_b']}",
        "",
        "Evidence currently available:"
    ]
    for i, ev in enumerate(evidence, 1):
        lines.append(f"{i}. {ev['text']}")
    lines.append("")
    lines.append("Which option is better supported by the evidence currently available?")
    return "\n".join(lines)

def parse_json_answer(text):
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.S)
        if not match:
            raise ValueError(f"Could not find JSON in model output: {text!r}")
        obj = json.loads(match.group(0))
    answer = str(obj["answer"]).upper().strip()
    confidence = float(obj["confidence"])
    if answer not in {"A", "B"}:
        raise ValueError(f"Invalid answer: {answer}")
    if not 0 <= confidence <= 1:
        raise ValueError(f"Confidence outside [0,1]: {confidence}")
    return answer, confidence

class MockProvider:
    """Pipeline validator only; not a research model."""
    def predict(self, case, evidence):
        score = {"A": 0.0, "B": 0.0}
        for ev in evidence:
            score[ev["stance"]] += float(ev["weight"])
        total = score["A"] + score["B"]
        if score["A"] == score["B"]:
            answer = evidence[-1]["stance"]
        else:
            answer = max(score, key=score.get)
        margin = abs(score["A"] - score["B"])
        confidence = 0.5 if total == 0 else min(0.99, 0.5 + 0.5 * margin / total)
        return answer, round(confidence, 4), {"raw": "mock-weighted-evidence"}

class OpenAIProvider:
    def __init__(self, model):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def predict(self, case, evidence):
        prompt = build_prompt(case, evidence)
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
        )
        raw = response.output_text
        answer, confidence = parse_json_answer(raw)
        return answer, confidence, {"raw": raw}

def get_provider(name, model):
    if name == "mock":
        return MockProvider()
    if name == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --provider openai")
        return OpenAIProvider(model)
    raise ValueError(f"Unknown provider: {name}")

def run_case(provider, case, order_name, order):
    ordered = [case["evidence"][i] for i in order]
    rows = []
    for step in range(1, len(ordered) + 1):
        visible = ordered[:step]
        answer, confidence, meta = provider.predict(case, visible)
        rows.append({
            "case_id": case["id"],
            "domain": case["domain"],
            "gold_answer": case["gold_answer"],
            "order_name": order_name,
            "order_evidence_ids": [ev["id"] for ev in ordered],
            "step": step,
            "visible_evidence_ids": [ev["id"] for ev in visible],
            "answer": answer,
            "confidence": confidence,
            "correct": answer == case["gold_answer"],
            "last_evidence_stance": visible[-1]["stance"],
            "last_evidence_weight": visible[-1]["weight"],
            "model_raw": meta.get("raw"),
        })
    return rows

def main():
    parser = argparse.ArgumentParser(description="Premature-closure / evidence-order benchmark")
    parser.add_argument("--dataset", default="dataset_v0.jsonl")
    parser.add_argument("--output", default="results/results.jsonl")
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default="gpt-5.6")
    parser.add_argument("--orders", nargs="*", default=list(ORDERS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[:args.limit]

    unknown = [o for o in args.orders if o not in ORDERS]
    if unknown:
        raise SystemExit(f"Unknown orders: {unknown}. Available: {list(ORDERS)}")

    provider = get_provider(args.provider, args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(cases) * len(args.orders) * 5
    done = 0
    with out_path.open("w", encoding="utf-8") as f:
        for case in cases:
            for order_name in args.orders:
                for row in run_case(provider, case, order_name, ORDERS[order_name]):
                    row["provider"] = args.provider
                    row["model"] = args.model if args.provider == "openai" else "mock"
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                    print(f"[{done}/{total}] {case['id']} {order_name} step={row['step']} -> {row['answer']} ({row['confidence']:.2f})")

    print(f"\nSaved {done} checkpoint predictions to {out_path}")
    print("Next: python analyze_results.py", out_path)

if __name__ == "__main__":
    main()
