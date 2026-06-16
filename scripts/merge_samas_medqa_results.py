"""Merge single-method SAMAS MedQA result directories into one table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    method_keys = [k for k in data if k.startswith("B")]
    if len(method_keys) != 1:
        raise ValueError(f"{path} should contain exactly one method, got {method_keys}")
    name = method_keys[0]
    medqa = data[name]["medqa"]
    raw = medqa.get("raw", [])
    errors = sum(1 for row in raw if str(row.get("raw_answer", "")).startswith("ERROR"))
    return {
        "method": name,
        "n": medqa["n"],
        "acc": medqa["accuracy"],
        "abstain": medqa["abstain_rate"],
        "answered_acc": medqa["correct_on_answered"],
        "avg_rounds": medqa["avg_rounds"],
        "avg_elapsed_s": medqa["avg_elapsed_s"],
        "wall_s": data[name]["wall_s"],
        "errors": errors,
        "path": str(path),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python scripts/merge_samas_medqa_results.py <results.json> ...")
        return 2
    rows = [summarize(Path(p)) for p in argv]
    rows.sort(key=lambda r: r["method"])
    print("| method | n | acc | abstain | answered_acc | avg_rounds | wall_s | errors |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['method']} | {r['n']} | {r['acc']:.3f} | {r['abstain']:.3f} | "
            f"{r['answered_acc']:.3f} | {r['avg_rounds']:.2f} | {r['wall_s']:.1f} | {r['errors']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
