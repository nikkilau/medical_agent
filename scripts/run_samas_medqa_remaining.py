"""Resume-capable MedQA baseline runner for SAMAS experiments.

This script intentionally lives under scripts/ and does not change SAMAS core.
It runs selected baseline methods into one directory per method and rewrites
results.json after each case, so interrupted full-MedQA sweeps can resume by
skipping completed case_id rows.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from medical_ttc.baselines import BaselineIOWorkflow, CoTSCWorkflow, extract_mcq_letter  # noqa: E402
from medical_ttc.data import LabeledCase, load_medqa  # noqa: E402
from medical_ttc.source_baselines import MedAgentsSourceWorkflow, MDAgentsSourceWorkflow  # noqa: E402
from medical_ttc.workflow import Workflow, WorkflowResult  # noqa: E402
from run_samas_matrix_medqa import (  # noqa: E402
    CoTWorkflow,
    MatrixConfig,
    make_embedding_provider,
    make_provider,
    make_samas_operator_workflow,
    parse_config,
)


DEFAULT_METHODS = [
    "B0_RawIO",
    "B1_CoT",
    "B2_CoTSC_5",
    "B3_MedAgents",
    "B4_MDAgents",
    "B5_SAMAS_ZS",
]


def requested_n() -> int | None:
    raw = os.environ.get("SAMAS_N_MEDQA", "").strip()
    if not raw or raw.lower() in {"all", "full", "none"}:
        return None
    return int(raw)


def shard_config() -> tuple[int, int]:
    count = int(os.environ.get("SAMAS_SHARD_COUNT", "1"))
    index = int(os.environ.get("SAMAS_SHARD_INDEX", "0"))
    if count < 1:
        raise ValueError("SAMAS_SHARD_COUNT must be >= 1")
    if index < 0 or index >= count:
        raise ValueError("SAMAS_SHARD_INDEX must satisfy 0 <= index < SAMAS_SHARD_COUNT")
    return count, index


def build_matrix(cfg: MatrixConfig, provider) -> dict[str, Callable[[], Workflow]]:
    embedding_provider = make_embedding_provider(cfg)
    return {
        "B0_RawIO": lambda: BaselineIOWorkflow(provider),
        "B1_CoT": lambda: CoTWorkflow(provider),
        "B2_CoTSC_5": lambda: CoTSCWorkflow(provider, n_samples=5),
        "B3_MedAgents": lambda: MedAgentsSourceWorkflow(
            provider,
            method=cfg.medagents_method,
            max_attempt_vote=cfg.medagents_max_attempt_vote,
        ),
        "B4_MDAgents": lambda: MDAgentsSourceWorkflow(
            provider,
            difficulty=cfg.mdagents_difficulty,
        ),
        "B5_SAMAS_ZS": lambda: make_samas_operator_workflow(provider, embedding_provider, cfg),
    }


def case_metric_row(lc: LabeledCase, out: WorkflowResult, elapsed: float) -> dict:
    predicted = extract_mcq_letter(out.final_answer)
    correct = predicted == lc.ground_truth and not out.abstained
    return {
        "case_id": lc.case.case_id,
        "correct": correct,
        "abstained": out.abstained,
        "rounds": out.rounds,
        "difficulty": out.difficulty,
        "expected_difficulty": str(lc.case.expected_difficulty),
        "elapsed_s": elapsed,
        "raw_answer": predicted,
        "ground_truth": lc.ground_truth,
    }


def error_metric_row(lc: LabeledCase, exc: Exception, elapsed: float) -> dict:
    return {
        "case_id": lc.case.case_id,
        "correct": False,
        "abstained": False,
        "rounds": 0,
        "difficulty": None,
        "expected_difficulty": str(lc.case.expected_difficulty),
        "elapsed_s": elapsed,
        "raw_answer": f"ERROR: {exc}",
        "ground_truth": lc.ground_truth,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    answered = [r for r in rows if not r.get("abstained")]
    return {
        "n": n,
        "accuracy": sum(1 for r in rows if r.get("correct")) / max(n, 1),
        "abstain_rate": sum(1 for r in rows if r.get("abstained")) / max(n, 1),
        "correct_on_answered": sum(1 for r in answered if r.get("correct")) / max(len(answered), 1),
        "avg_rounds": statistics.mean(float(r.get("rounds") or 0) for r in rows) if rows else 0.0,
        "avg_elapsed_s": statistics.mean(float(r.get("elapsed_s") or 0.0) for r in rows) if rows else 0.0,
        "score_g": sum(1 for r in rows if r.get("correct")) / max(n, 1),
        "raw": rows,
    }


def save_result(path: Path, cfg: MatrixConfig, n_limit: int | None, method: str, rows: list[dict], wall_s: float) -> None:
    data = {
        "config": {
            **asdict(cfg),
            "n_medqa_requested": n_limit if n_limit is not None else "full",
            "api_key_env": cfg.api_key_env,
            "api_key_present": True,
        },
        "started": time.time() - wall_s,
        method: {
            "medqa": aggregate(rows),
            "wall_s": wall_s,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_rows(path: Path, method: str, valid_ids: set[str], retry_errors: bool) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get(method, {}).get("medqa", {}).get("raw", [])
    filtered = []
    for row in rows:
        if row.get("case_id") not in valid_ids:
            continue
        if retry_errors and str(row.get("raw_answer", "")).startswith("ERROR"):
            continue
        filtered.append(row)
    return filtered


def result_path_for(out_root: Path, method: str, shard_count: int, shard_index: int) -> Path:
    if shard_count <= 1:
        return out_root / method / "results.json"
    return out_root / method / "shards" / f"shard_{shard_index:02d}_of_{shard_count:02d}.json"


def run_method(
    method: str,
    factory: Callable[[], Workflow],
    cases: list[LabeledCase],
    out_root: Path,
    cfg: MatrixConfig,
    n_limit: int | None,
    *,
    shard_count: int = 1,
    shard_index: int = 0,
) -> int:
    result_path = result_path_for(out_root, method, shard_count, shard_index)
    valid_ids = {lc.case.case_id for lc in cases}
    retry_errors = os.environ.get("SAMAS_RETRY_ERRORS", "").strip() == "1"
    primary_path = out_root / method / "results.json"
    rows_by_id = {
        row.get("case_id"): row
        for row in load_rows(primary_path, method, valid_ids, retry_errors)
    }
    rows_by_id.update(
        {
            row.get("case_id"): row
            for row in load_rows(result_path, method, valid_ids, retry_errors)
        }
    )
    rows = [row for row in rows_by_id.values() if row.get("case_id")]
    completed = {r.get("case_id") for r in rows}
    remaining = [lc for lc in cases if lc.case.case_id not in completed]
    shard_note = f" shard={shard_index}/{shard_count}" if shard_count > 1 else ""
    print(f"[{method}] completed={len(rows)} remaining={len(remaining)}{shard_note} out={result_path}", flush=True)

    started = time.time()
    workflow = factory()
    for idx, lc in enumerate(remaining, start=1):
        t0 = time.time()
        try:
            out = workflow(lc.case)
            row = case_metric_row(lc, out, time.time() - t0)
        except Exception as exc:
            row = error_metric_row(lc, exc, time.time() - t0)
        rows.append(row)
        save_result(result_path, cfg, n_limit, method, rows, time.time() - started)
        status = "OK" if row["correct"] else ("ABSTAIN" if row["abstained"] else "MISS")
        print(
            f"[{method}] {len(rows)}/{len(cases)} (+{idx}/{len(remaining)}) "
            f"{status} pred={row['raw_answer']} gt={row['ground_truth']} t={row['elapsed_s']:.1f}s",
            flush=True,
        )

    save_result(result_path, cfg, n_limit, method, rows, time.time() - started)
    metrics = aggregate(rows)
    print(
        f"[{method}] done n={metrics['n']} acc={metrics['accuracy']:.3f} "
        f"abstain={metrics['abstain_rate']:.3f} wall={time.time() - started:.1f}s",
        flush=True,
    )
    return 0


def main() -> int:
    cfg = parse_config()
    n_limit = requested_n()
    shard_count, shard_index = shard_config()
    out_root = Path(os.environ.get("SAMAS_OUT", "reports/samas_baselines_medqa_full_source"))
    methods = [
        m.strip()
        for m in os.environ.get("SAMAS_METHODS", ",".join(DEFAULT_METHODS)).split(",")
        if m.strip()
    ]

    print(f"[data] loading MedQA split={cfg.split} n={n_limit if n_limit is not None else 'full'} seed={cfg.seed}")
    all_cases = load_medqa(split=cfg.split, n=n_limit, seed=cfg.seed)
    if shard_count > 1:
        cases = [lc for i, lc in enumerate(all_cases) if i % shard_count == shard_index]
        print(f"[data] got {len(all_cases)} cases; shard {shard_index}/{shard_count} has {len(cases)} cases")
    else:
        cases = all_cases
        print(f"[data] got {len(cases)} cases")

    provider = make_provider(cfg)
    matrix = build_matrix(cfg, provider)
    unknown = sorted(set(methods) - set(matrix))
    if unknown:
        raise ValueError(f"Unsupported baseline methods for this runner: {unknown}")

    for method in methods:
        run_method(
            method,
            matrix[method],
            cases,
            out_root,
            cfg,
            n_limit,
            shard_count=shard_count,
            shard_index=shard_index,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
