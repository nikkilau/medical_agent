"""Run selected SAMAS MedQA methods concurrently.

This wrapper is useful for OpenAI-compatible endpoints that support multiple
in-flight requests. Each method writes its own result directory, then a merged
Markdown table is printed at the end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_METHODS = [
    "B2_CoTSC_5",
    "B3_MedAgents",
    "B4_MDAgents",
    "B5_SAMAS_ZS",
]


def run_method(method: str, *, n: int, out_root: Path) -> tuple[str, int, str]:
    out_dir = out_root / method
    env = os.environ.copy()
    env.update(
        {
            "SAMAS_N_MEDQA": str(n),
            "SAMAS_METHODS": method,
            "SAMAS_OUT": str(out_dir),
            "PYTHONPATH": "src",
        }
    )
    cmd = [sys.executable, "scripts/run_samas_matrix_medqa.py"]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return method, proc.returncode, proc.stdout


def summarize(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    methods = [k for k in data if k.startswith("B")]
    if len(methods) != 1:
        raise ValueError(f"{path} should have one method, got {methods}")
    method = methods[0]
    medqa = data[method]["medqa"]
    errors = sum(
        1 for row in medqa.get("raw", [])
        if str(row.get("raw_answer", "")).startswith("ERROR")
    )
    return {
        "method": method,
        "n": medqa["n"],
        "acc": medqa["accuracy"],
        "abstain": medqa["abstain_rate"],
        "answered_acc": medqa["correct_on_answered"],
        "avg_rounds": medqa["avg_rounds"],
        "wall_s": data[method]["wall_s"],
        "errors": errors,
    }


def main() -> int:
    n = int(os.environ.get("SAMAS_N_MEDQA", "20"))
    out_root = Path(os.environ.get("SAMAS_OUT", "reports/samas_matrix_medqa_parallel"))
    methods = [
        m.strip()
        for m in os.environ.get("SAMAS_METHODS", ",".join(DEFAULT_METHODS)).split(",")
        if m.strip()
    ]
    workers = int(os.environ.get("SAMAS_WORKERS", str(min(len(methods), 8))))
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[parallel] methods={methods} n={n} workers={workers} out={out_root}", flush=True)
    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_method, method, n=n, out_root=out_root) for method in methods]
        for fut in as_completed(futs):
            method, code, output = fut.result()
            log_path = out_root / method / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(output, encoding="utf-8")
            print(f"[parallel] {method} exit={code} log={log_path}", flush=True)
            if code != 0:
                failures.append((method, code))

    result_paths = [out_root / method / "results.json" for method in methods]
    rows = [summarize(path) for path in result_paths if path.exists()]
    rows.sort(key=lambda r: r["method"])

    print("\n| method | n | acc | abstain | answered_acc | avg_rounds | wall_s | errors |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['method']} | {r['n']} | {r['acc']:.3f} | {r['abstain']:.3f} | "
            f"{r['answered_acc']:.3f} | {r['avg_rounds']:.2f} | {r['wall_s']:.1f} | {r['errors']} |"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
