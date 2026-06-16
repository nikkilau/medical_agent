"""Run the SAMAS plan's 7-method matrix on MedQA only.

This is a pilot runner for the current plan scope:
- Profiler is prompt-only.
- Dataset is MedQA only.
- B6 trained-profiler is not claimed; it is reported as the prompt-profiler
  review variant until trained profiler/controller are implemented.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medical_ttc.baselines import (  # noqa: E402
    BaselineIOWorkflow,
    CoTSCWorkflow,
)
from medical_ttc.data import load_medqa  # noqa: E402
from medical_ttc.evaluator import AggregateMetrics, evaluate_medqa  # noqa: E402
from medical_ttc.operators import GenerateOperator  # noqa: E402
from medical_ttc.source_baselines import MedAgentsSourceWorkflow, MDAgentsSourceWorkflow  # noqa: E402
from medical_ttc.workflow import Workflow, WorkflowResult  # noqa: E402
from micro_mdt.models import PatientCase  # noqa: E402
from micro_mdt.providers import MockProvider, OpenAICompatibleEmbeddingProvider, OpenAICompatibleProvider  # noqa: E402
from samas import SAMASOperatorController  # noqa: E402
from samas.workflow import SAMASOperatorWorkflow, SAMASPromptWorkflow  # noqa: E402


@dataclass
class MatrixConfig:
    n_medqa: int = 20
    seed: int = 42
    split: str = "test"
    output_dir: str = "reports/samas_matrix_medqa"
    use_mock: bool = False
    base_url: str = "http://202.120.24.199:13000/v1"
    model: str = "qwen2.5:14b"
    embedding_model: str = "nomic-embed-text"
    api_key_env: str = "GPUSTACK_API_KEY"
    claude_path: str = "/home/sjtu/workspace/ycy/CLAUDE.md"
    methods: str = ""
    include_untrained_b6: bool = False
    medagents_method: str = "syn_verif"
    medagents_max_attempt_vote: int = 3
    mdagents_difficulty: str = "adaptive"
    samas_operator_checkpoint: str = ""
    samas_semantic_dim: int = 32


class CoTWorkflow(Workflow):
    def __init__(self, provider) -> None:
        super().__init__("B1_CoT")
        prompt = """You are a careful medical exam solver.
Think step by step, then choose exactly one option.
End with exactly: Final answer: <A/B/C/D>
"""
        self.generate = GenerateOperator(provider, prompt=prompt)

    def __call__(self, case: PatientCase) -> WorkflowResult:
        answer = self.generate(case.text, temperature=0.0)
        return WorkflowResult(
            case=case,
            final_answer=answer,
            rounds=1,
            difficulty="N/A",
            trace=[{"step": "cot", "answer": answer[:200]}],
        )


def parse_config() -> MatrixConfig:
    cfg = MatrixConfig()
    if os.environ.get("SAMAS_N_MEDQA"):
        cfg.n_medqa = int(os.environ["SAMAS_N_MEDQA"])
    if os.environ.get("SAMAS_SEED"):
        cfg.seed = int(os.environ["SAMAS_SEED"])
    if os.environ.get("SAMAS_MEDQA_SPLIT"):
        cfg.split = os.environ["SAMAS_MEDQA_SPLIT"]
    if os.environ.get("SAMAS_OUT"):
        cfg.output_dir = os.environ["SAMAS_OUT"]
    if os.environ.get("SAMAS_BASE_URL"):
        cfg.base_url = os.environ["SAMAS_BASE_URL"]
    if os.environ.get("SAMAS_MODEL"):
        cfg.model = os.environ["SAMAS_MODEL"]
    if os.environ.get("SAMAS_EMBEDDING_MODEL"):
        cfg.embedding_model = os.environ["SAMAS_EMBEDDING_MODEL"]
    if os.environ.get("SAMAS_API_KEY_ENV"):
        cfg.api_key_env = os.environ["SAMAS_API_KEY_ENV"]
    if os.environ.get("SAMAS_USE_MOCK"):
        cfg.use_mock = True
    if os.environ.get("SAMAS_METHODS"):
        cfg.methods = os.environ["SAMAS_METHODS"]
    if os.environ.get("SAMAS_INCLUDE_UNTRAINED_B6"):
        cfg.include_untrained_b6 = True
    if os.environ.get("SAMAS_MEDAGENTS_METHOD"):
        cfg.medagents_method = os.environ["SAMAS_MEDAGENTS_METHOD"]
    if os.environ.get("SAMAS_MEDAGENTS_MAX_ATTEMPT_VOTE"):
        cfg.medagents_max_attempt_vote = int(os.environ["SAMAS_MEDAGENTS_MAX_ATTEMPT_VOTE"])
    if os.environ.get("SAMAS_MDAGENTS_DIFFICULTY"):
        cfg.mdagents_difficulty = os.environ["SAMAS_MDAGENTS_DIFFICULTY"]
    if os.environ.get("SAMAS_OPERATOR_CONTROLLER_PATH"):
        cfg.samas_operator_checkpoint = os.environ["SAMAS_OPERATOR_CONTROLLER_PATH"]
    if os.environ.get("SAMAS_SEMANTIC_DIM"):
        cfg.samas_semantic_dim = int(os.environ["SAMAS_SEMANTIC_DIM"])
    return cfg


def read_api_key(cfg: MatrixConfig) -> str:
    env_key = os.environ.get(cfg.api_key_env)
    if env_key:
        return env_key
    path = Path(cfg.claude_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"api_key:\s*(sk-[^\s]+)", text)
    return match.group(1) if match else ""


def make_provider(cfg: MatrixConfig):
    api_key = read_api_key(cfg)
    if cfg.use_mock or not api_key:
        print("[provider] using MockProvider")
        return MockProvider()
    print(f"[provider] using OpenAI-compatible endpoint {cfg.base_url}/v1 model={cfg.model}")
    return OpenAICompatibleProvider(
        base_url=f"{cfg.base_url}/v1" if not cfg.base_url.endswith("/v1") else cfg.base_url,
        model=cfg.model,
        api_key=api_key,
        timeout=180,
    )


def make_embedding_provider(cfg: MatrixConfig):
    api_key = read_api_key(cfg)
    if cfg.use_mock or not api_key:
        return None
    print(f"[embedding] using OpenAI-compatible endpoint {cfg.base_url}/v1 model={cfg.embedding_model}")
    return OpenAICompatibleEmbeddingProvider(
        base_url=f"{cfg.base_url}/v1" if not cfg.base_url.endswith("/v1") else cfg.base_url,
        model=cfg.embedding_model,
        api_key=api_key,
        timeout=180,
    )


def run_one(name: str, factory: Callable[[], Workflow], cases: list) -> dict:
    print(f"\n=== {name} ({len(cases)} MedQA cases) ===", flush=True)
    workflow = factory()
    t0 = time.time()
    metrics = evaluate_medqa(workflow, cases, verbose=True)
    elapsed = time.time() - t0
    print(
        f"[{name}] acc={metrics.accuracy:.3f} "
        f"abstain={metrics.abstain_rate:.3f} "
        f"answered_acc={metrics.correct_on_answered:.3f} "
        f"avg_rounds={metrics.avg_rounds:.2f} "
        f"wall={elapsed:.1f}s",
        flush=True,
    )
    return {"medqa": metrics.to_dict(), "wall_s": elapsed}


def make_samas_operator_workflow(provider, embedding_provider, cfg: MatrixConfig) -> SAMASOperatorWorkflow:
    controller = None
    if cfg.samas_operator_checkpoint:
        path = Path(cfg.samas_operator_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"SAMAS operator controller checkpoint not found: {path}")
        controller = SAMASOperatorController.load(path)
    else:
        controller = SAMASOperatorController(semantic_dim=cfg.samas_semantic_dim)
    return SAMASOperatorWorkflow(
        provider,
        controller=controller,
        embedding_provider=embedding_provider,
        sample=False,
    )


def main() -> int:
    cfg = parse_config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading MedQA split={cfg.split} n={cfg.n_medqa} seed={cfg.seed}")
    cases = load_medqa(split=cfg.split, n=cfg.n_medqa, seed=cfg.seed)
    print(f"[data] got {len(cases)} cases")

    provider = make_provider(cfg)
    embedding_provider = make_embedding_provider(cfg)
    started = time.time()
    results: dict = {
        "config": {
            **asdict(cfg),
            "api_key_env": cfg.api_key_env,
            "api_key_present": bool(read_api_key(cfg)),
        },
        "started": started,
        "notes": {
            "B6": "Not run in this pilot. B6 requires trained profiler/controller and must not be reported from prompt-only SAMAS.",
        },
    }

    matrix: list[tuple[str, Callable[[], Workflow]]] = [
        ("B0_RawIO", lambda: BaselineIOWorkflow(provider)),
        ("B1_CoT", lambda: CoTWorkflow(provider)),
        ("B2_CoTSC_5", lambda: CoTSCWorkflow(provider, n_samples=5)),
        (
            "B3_MedAgents",
            lambda: MedAgentsSourceWorkflow(
                provider,
                method=cfg.medagents_method,
                max_attempt_vote=cfg.medagents_max_attempt_vote,
            ),
        ),
        (
            "B4_MDAgents",
            lambda: MDAgentsSourceWorkflow(
                provider,
                difficulty=cfg.mdagents_difficulty,
            ),
        ),
        ("B5_SAMAS_ZS", lambda: make_samas_operator_workflow(provider, embedding_provider, cfg)),
    ]
    if cfg.include_untrained_b6:
        matrix.append(
            (
                "B6_UNTRAINED_PROMPT_REVIEW_DO_NOT_REPORT_AS_B6",
                lambda: SAMASPromptWorkflow(provider, enable_review=True),
            )
        )
    if cfg.methods:
        wanted = {m.strip() for m in cfg.methods.split(",") if m.strip()}
        invalid_b6 = {m for m in wanted if m.startswith("B6")} and not cfg.include_untrained_b6
        if invalid_b6:
            raise ValueError(
                "B6 requires trained profiler/controller and is not available in the prompt-only pilot. "
                "Set SAMAS_INCLUDE_UNTRAINED_B6=1 only for debugging, not for reported results."
            )
        matrix = [(name, factory) for name, factory in matrix if name in wanted]
        missing = wanted - {name for name, _ in matrix}
        if missing:
            raise ValueError(f"Unknown SAMAS_METHODS entries: {sorted(missing)}")

    for name, factory in matrix:
        results[name] = run_one(name, factory, cases)
        with (out_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    results["elapsed_s"] = time.time() - started
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nmethod,acc,abstain,answered_acc,avg_rounds,wall_s")
    for name, _ in matrix:
        metrics = AggregateMetrics(**{k: v for k, v in results[name]["medqa"].items() if k != "raw"})
        print(
            f"{name},{metrics.accuracy:.3f},{metrics.abstain_rate:.3f},"
            f"{metrics.correct_on_answered:.3f},{metrics.avg_rounds:.2f},"
            f"{results[name]['wall_s']:.1f}"
        )
    print(f"[done] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
