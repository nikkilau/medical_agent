"""Phase 3 follow-up: run AFLOW MCTS search + W7 DSPy alone.

Assumes baseline pilot results already live in `--baseline-dir`.
Outputs MCTS history + W4_searched + W7_DSPy_PromptOnly into a new
results.json and merges with the baseline numbers when generating the report.

Env vars:
    DASHSCOPE_API_KEY   (required) Aliyun key for qwen3.6-plus optimizer
    HF_ENDPOINT         HuggingFace mirror (default hf-mirror.com)
    MEDTTC_MCTS_ITERS   number of MCTS iterations (default 6)
    MEDTTC_MCTS_VALN    validation cases per iteration (default 20)
    MEDTTC_N_MEDQA      MedQA cases for final test (default 30)
    MEDTTC_N_SAFETY     MedSafetyBench cases (default 10)
    MEDTTC_OUT          output dir (default reports/mcts_default)
    MEDTTC_BASELINE_RESULTS  path to baseline results.json to merge into final report
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medical_ttc.data import load_medqa, load_medsafety  # noqa: E402
from medical_ttc.evaluator import (  # noqa: E402
    compute_g_score,
    evaluate_medqa,
    evaluate_medsafety,
)
from medical_ttc.mcts import MCTSConfig, run_mcts  # noqa: E402
from medical_ttc.router import HeuristicRouter  # noqa: E402
from medical_ttc.ttc_workflow import MedicalTTCWorkflow  # noqa: E402
from medical_ttc.dspy_baseline import (  # noqa: E402
    DSPyPromptOnlyWorkflow,
    run_prompt_grid_search,
)
from micro_mdt.providers import OpenAICompatibleProvider  # noqa: E402


def main() -> int:
    out_dir = Path(os.environ.get("MEDTTC_OUT", "reports/mcts_default"))
    out_dir.mkdir(parents=True, exist_ok=True)

    n_medqa = int(os.environ.get("MEDTTC_N_MEDQA", "30"))
    n_safety = int(os.environ.get("MEDTTC_N_SAFETY", "10"))
    n_mcts_iter = int(os.environ.get("MEDTTC_MCTS_ITERS", "6"))
    n_val = int(os.environ.get("MEDTTC_MCTS_VALN", "20"))
    baseline_path = os.environ.get("MEDTTC_BASELINE_RESULTS", "")

    print(f"MCTS config: iters={n_mcts_iter} val_n={n_val} medqa={n_medqa} safety={n_safety}")

    # Datasets
    medqa = load_medqa(split="test", n=n_medqa, seed=42)
    medsafety = load_medsafety(n=n_safety, seed=42)
    val_cases = medqa[:n_val]
    # Test set: ideally disjoint from val. For pilot we accept overlap when
    # n_medqa <= n_val. Always at least evaluate on the full n_medqa.
    test_cases = medqa[n_val:] if len(medqa) > n_val else medqa

    # Providers
    executor = OpenAICompatibleProvider(
        base_url="http://202.120.5.12:18080/v1",
        model="qwen2.5:14b",
        api_key=os.environ.get("GPUSTACK_API_KEY", "gpustack_4ba10aa3dd610a36_843f703126bc1aa7b9c05c7486a388be"),
    )
    optimizer = OpenAICompatibleProvider(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.6-plus",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )

    record: dict = {
        "phase": "Phase 3: MCTS + W7",
        "config": {
            "n_medqa": n_medqa,
            "n_safety": n_safety,
            "mcts_iterations": n_mcts_iter,
            "mcts_validation_n": n_val,
            "executor_model": "qwen2.5:14b",
            "optimizer_model": "qwen3.6-plus",
        },
        "started": time.time(),
    }

    # --- W7 prompt grid search ---
    print("\n=== W7 DSPy-style prompt grid search ===")
    t0 = time.time()
    grid = run_prompt_grid_search(
        executor, optimizer, val_cases[: min(15, n_val)], n_candidates=3
    )
    record["w7_grid_scores"] = grid.scores
    record["w7_best_prompt"] = grid.best_prompt[:600]
    record["w7_grid_wall_s"] = time.time() - t0
    print(f"  done in {record['w7_grid_wall_s']:.0f}s; scores={grid.scores}")

    print("=== W7 final evaluation on test split ===")
    w7_wf = DSPyPromptOnlyWorkflow(executor, grid.best_prompt)
    t1 = time.time()
    w7_med = evaluate_medqa(w7_wf, test_cases)
    w7_safe = evaluate_medsafety(w7_wf, medsafety)
    record["W7_DSPy_PromptOnly"] = {
        "medqa": w7_med.to_dict(),
        "medsafety": w7_safe,
        "g_score": compute_g_score(w7_med, w7_safe),
        "wall_s": time.time() - t1,
    }
    print(
        f"  W7: acc={w7_med.accuracy:.3f} unsafe={w7_safe['unsafe_rate']:.3f} "
        f"G={record['W7_DSPy_PromptOnly']['g_score']:.3f}"
    )

    # --- AFLOW MCTS over W4 prompts ---
    print(f"\n=== W4* MCTS search ({n_mcts_iter} iter, val_n={n_val}) ===")
    mcts_cfg = MCTSConfig(
        n_iterations=n_mcts_iter,
        top_k=3,
        early_stop_patience=3,
        eval_n_validation=n_val,
        n_eval_repeats=1,
        log_dir=str(out_dir / "mcts"),
        seed=42,
    )

    def factory(p):
        return MedicalTTCWorkflow(
            executor, router=HeuristicRouter(), prompts=p
        )

    best, history = run_mcts(
        executor, optimizer, val_cases, config=mcts_cfg, workflow_factory=factory
    )
    record["mcts_history"] = [n.to_dict() for n in history]
    record["mcts_best_node_id"] = best.node_id
    record["mcts_best_validation_score"] = best.score

    # --- W4_searched on test split ---
    print("=== W4* (MCTS-best) evaluation on test split ===")
    w4_star = MedicalTTCWorkflow(
        executor, router=HeuristicRouter(), prompts=best.prompts, name="W4_MedicalTTC_searched"
    )
    t2 = time.time()
    w4s_med = evaluate_medqa(w4_star, test_cases)
    w4s_safe = evaluate_medsafety(w4_star, medsafety)
    record["W4_MedicalTTC_searched"] = {
        "medqa": w4s_med.to_dict(),
        "medsafety": w4s_safe,
        "g_score": compute_g_score(w4s_med, w4s_safe),
        "wall_s": time.time() - t2,
    }
    print(
        f"  W4*: acc={w4s_med.accuracy:.3f} unsafe={w4s_safe['unsafe_rate']:.3f} "
        f"G={record['W4_MedicalTTC_searched']['g_score']:.3f}"
    )

    record["ended"] = time.time()

    # Merge with baseline results if path provided
    if baseline_path and Path(baseline_path).exists():
        base = json.loads(Path(baseline_path).read_text())
        for k, v in base.items():
            if k not in record and k not in {"started", "ended"}:
                record[k] = v
        record["merged_from"] = baseline_path

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
