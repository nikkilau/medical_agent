"""Run the full Medical TTC experiment matrix.

Usage:
    PYTHONPATH=src python scripts/run_all_experiments.py

Configured for the SJTU server: GPUstack qwen2.5:14b as executor, qwen3.6-plus
as MCTS optimizer.

Phase 4 pilot defaults: 50 MedQA + 10 MedSafetyBench; 3 MCTS iterations.
Set MEDTTC_FULL=1 to scale to 200 MedQA + 30 MedSafetyBench + 10 MCTS iterations.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

# Allow `python scripts/...` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medical_ttc.baselines import (  # noqa: E402
    BaselineIOWorkflow,
    CoTSCWorkflow,
    HandCodedMicroMDTWorkflow,
)
from medical_ttc.data import load_medqa, load_medsafety  # noqa: E402
from medical_ttc.evaluator import (  # noqa: E402
    AggregateMetrics,
    compute_g_score,
    evaluate_medqa,
    evaluate_medsafety,
)
from medical_ttc.mcts import MCTSConfig, run_mcts  # noqa: E402
from medical_ttc.source_baselines import MDAgentsSourceWorkflow  # noqa: E402
from medical_ttc.ttc_workflow import MedicalTTCWorkflow, TierPrompts  # noqa: E402
from medical_ttc.router import ZeroShotRouter, HeuristicRouter  # noqa: E402
from micro_mdt.providers import (  # noqa: E402
    MockProvider,
    OpenAICompatibleProvider,
)


# --------------------------- Config ------------------------------


@dataclass
class ExperimentConfig:
    n_medqa: int = 50
    n_medsafety: int = 10
    mcts_iterations: int = 3
    mcts_validation_n: int = 25
    use_mock: bool = False
    executor_base_url: str = "http://202.120.5.12:18080/v1"
    executor_model: str = "qwen2.5:14b"
    executor_api_key: str = "gpustack_4ba10aa3dd610a36_843f703126bc1aa7b9c05c7486a388be"
    optimizer_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    optimizer_model: str = "qwen3.6-plus"
    optimizer_api_key: str = os.environ.get("DASHSCOPE_API_KEY", "")
    output_dir: str = "reports/run_default"
    seed: int = 42
    skip_mcts: bool = False  # skip the W4 search (cheap path for first smoke run)


def parse_config() -> ExperimentConfig:
    cfg = ExperimentConfig()
    if os.environ.get("MEDTTC_FULL"):
        cfg.n_medqa = 200
        cfg.n_medsafety = 30
        cfg.mcts_iterations = 10
        cfg.mcts_validation_n = 50
    if os.environ.get("MEDTTC_USE_MOCK"):
        cfg.use_mock = True
    if os.environ.get("MEDTTC_SKIP_MCTS"):
        cfg.skip_mcts = True
    if os.environ.get("MEDTTC_N_MEDQA"):
        cfg.n_medqa = int(os.environ["MEDTTC_N_MEDQA"])
    if os.environ.get("MEDTTC_N_SAFETY"):
        cfg.n_medsafety = int(os.environ["MEDTTC_N_SAFETY"])
    if os.environ.get("MEDTTC_MCTS_ITERS"):
        cfg.mcts_iterations = int(os.environ["MEDTTC_MCTS_ITERS"])
    if os.environ.get("MEDTTC_MCTS_VALN"):
        cfg.mcts_validation_n = int(os.environ["MEDTTC_MCTS_VALN"])
    out = os.environ.get("MEDTTC_OUT")
    if out:
        cfg.output_dir = out
    api = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIYUN_API_KEY")
    if api:
        cfg.optimizer_api_key = api
    return cfg


# --------------------------- Providers ------------------------------


def make_executor_provider(cfg: ExperimentConfig):
    if cfg.use_mock:
        return MockProvider()
    return OpenAICompatibleProvider(
        base_url=cfg.executor_base_url,
        model=cfg.executor_model,
        api_key=cfg.executor_api_key,
    )


def make_optimizer_provider(cfg: ExperimentConfig):
    if cfg.use_mock or not cfg.optimizer_api_key:
        return MockProvider()
    return OpenAICompatibleProvider(
        base_url=cfg.optimizer_base_url,
        model=cfg.optimizer_model,
        api_key=cfg.optimizer_api_key,
    )


# --------------------------- Experiment runners ---------------------


def run_one(
    name: str,
    workflow_factory: Callable[[], object],
    medqa_cases: list,
    medsafety_cases: list,
    *,
    record: dict,
    skip_safety: bool = False,
) -> None:
    print(f"\n=== {name} on MedQA ({len(medqa_cases)} cases) ===")
    wf = workflow_factory()
    t0 = time.time()
    med_metrics = evaluate_medqa(wf, medqa_cases, verbose=False)
    t_med = time.time() - t0
    if skip_safety or not medsafety_cases:
        safety = {"unsafe_rate": 0.0, "n": 0, "raw": []}
    else:
        print(f"=== {name} on MedSafetyBench ({len(medsafety_cases)} cases) ===")
        t1 = time.time()
        safety = evaluate_medsafety(wf, medsafety_cases, verbose=False)
        safety["wall_s"] = time.time() - t1
    g = compute_g_score(med_metrics, safety)
    print(
        f"-- {name}: acc={med_metrics.accuracy:.3f} "
        f"abstain={med_metrics.abstain_rate:.3f} "
        f"unsafe={safety['unsafe_rate']:.3f} G={g:.3f} ({t_med:.0f}s)"
    )
    record[name] = {
        "medqa": med_metrics.to_dict(),
        "medsafety": safety,
        "g_score": g,
        "wall_s_medqa": t_med,
    }


def main() -> int:
    cfg = parse_config()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading MedQA ({cfg.n_medqa})…")
    medqa = load_medqa(split="test", n=cfg.n_medqa, seed=cfg.seed)
    print(f"  got {len(medqa)} cases")
    print(f"Loading MedSafetyBench ({cfg.n_medsafety})…")
    medsafety = load_medsafety(n=cfg.n_medsafety, seed=cfg.seed)
    print(f"  got {len(medsafety)} cases")

    executor = make_executor_provider(cfg)
    optimizer = make_optimizer_provider(cfg)

    record: dict = {"config": asdict(cfg), "started": time.time()}

    # --- W0 BaselineIO ---
    run_one("W0_BaselineIO", lambda: BaselineIOWorkflow(executor), medqa, medsafety, record=record)
    # --- W1 CoT-SC ---
    run_one("W1_CoTSC", lambda: CoTSCWorkflow(executor, n_samples=3), medqa, medsafety, record=record)
    # --- W2 Hand-Coded Micro-MDT ---
    run_one("W2_HandCoded", lambda: HandCodedMicroMDTWorkflow(executor), medqa, medsafety, record=record)
    # --- W3 MDAgents source baseline ---
    run_one(
        "W3_MDAgents_Source",
        lambda: MDAgentsSourceWorkflow(executor, difficulty="adaptive"),
        medqa,
        medsafety,
        record=record,
    )
    # --- W4 Medical TTC (default prompts) ---
    run_one(
        "W4_MedicalTTC_default",
        lambda: MedicalTTCWorkflow(executor, router=HeuristicRouter()),
        medqa,
        medsafety,
        record=record,
    )
    # --- W5 W4 without router (everything goes HIGH) ---
    class _AllHighRouter:
        def route(self, _):
            from micro_mdt.models import Difficulty
            return Difficulty.HIGH
    run_one(
        "W5_W4_no_router",
        lambda: MedicalTTCWorkflow(executor, router=_AllHighRouter()),
        medqa,
        medsafety,
        record=record,
    )
    # --- W6 W4 without safety hard constraint ---
    run_one(
        "W6_W4_no_safety",
        lambda: MedicalTTCWorkflow(executor, router=HeuristicRouter(), enable_safety=False),
        medqa,
        medsafety,
        record=record,
    )

    # --- W7 DSPy-style prompt-only optimization (cheap prompt grid search) ---
    if not cfg.skip_mcts:
        from medical_ttc.dspy_baseline import (
            DSPyPromptOnlyWorkflow,
            run_prompt_grid_search,
        )
        print(f"\n=== W7 prompt grid search ===")
        val_cases = medqa[: min(15, cfg.mcts_validation_n)]
        grid = run_prompt_grid_search(
            executor, optimizer, val_cases, n_candidates=3
        )
        record["w7_grid"] = {
            "scores": grid.scores,
            "best_index": int(grid.scores.index(max(grid.scores))),
            "best_prompt": grid.best_prompt[:600],
        }
        w7_wf = DSPyPromptOnlyWorkflow(executor, grid.best_prompt)
        run_one(
            "W7_DSPy_PromptOnly",
            lambda: w7_wf,
            medqa,
            medsafety,
            record=record,
        )

    # --- W4* Medical TTC after MCTS search ---
    if not cfg.skip_mcts:
        print(f"\n=== W4* MCTS search ({cfg.mcts_iterations} iter, n_val={cfg.mcts_validation_n}) ===")
        val_cases = medqa[: cfg.mcts_validation_n]
        mcts_cfg = MCTSConfig(
            n_iterations=cfg.mcts_iterations,
            top_k=3,
            early_stop_patience=4,
            eval_n_validation=cfg.mcts_validation_n,
            n_eval_repeats=1,
            log_dir=os.path.join(cfg.output_dir, "mcts"),
            seed=cfg.seed,
        )
        best, history = run_mcts(
            executor,
            optimizer,
            val_cases,
            config=mcts_cfg,
        )
        record["mcts_history"] = [n.to_dict() for n in history]
        record["mcts_best_node_id"] = best.node_id
        record["mcts_best_score_on_validation"] = best.score
        # Evaluate the best workflow on the held-out test set
        searched_workflow = MedicalTTCWorkflow(
            executor, router=HeuristicRouter(), prompts=best.prompts, name="W4_searched"
        )
        run_one(
            "W4_MedicalTTC_searched",
            lambda: searched_workflow,
            medqa[cfg.mcts_validation_n :] or medqa,
            medsafety,
            record=record,
        )

    record["ended"] = time.time()
    out_path = Path(cfg.output_dir) / "results.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
