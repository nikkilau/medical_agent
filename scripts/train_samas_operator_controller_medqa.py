"""Train the SAMAS step-wise operator controller on MedQA.

This is the main MaAS-style controller path:
PromptProfiler-ZS -> SAMASOperatorController -> execute sampled operator plan
-> MedQA reward -> REINFORCE update.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medical_ttc.baselines import extract_mcq_letter  # noqa: E402
from medical_ttc.data import LabeledCase, load_medqa  # noqa: E402
from medical_ttc.evaluator import evaluate_medqa  # noqa: E402
from medical_ttc.safety import assess_response_safety  # noqa: E402
from micro_mdt.providers import MockProvider, OpenAICompatibleEmbeddingProvider, OpenAICompatibleProvider  # noqa: E402
from samas import SAMASOperatorController, SAMASOperatorWorkflow  # noqa: E402


@dataclass
class TrainConfig:
    train_n: int = 20
    eval_n: int = 20
    train_split: str = "train"
    eval_split: str = "test"
    seed: int = 42
    epochs: int = 1
    max_steps: int = 6
    semantic_dim: int = 32
    output_dir: str = "reports/samas_operator_controller_medqa"
    checkpoint: str = "reports/samas_operator_controller_medqa/controller.pt"
    use_mock: bool = False
    base_url: str = "http://202.120.24.199:13000/v1"
    model: str = "qwen2.5:14b"
    embedding_model: str = "nomic-embed-text"
    api_key_env: str = "GPUSTACK_API_KEY"
    claude_path: str = "/home/sjtu/workspace/ycy/CLAUDE.md"


def parse_config() -> TrainConfig:
    cfg = TrainConfig()
    env_map = {
        "SAMAS_TRAIN_N": ("train_n", int),
        "SAMAS_EVAL_N": ("eval_n", int),
        "SAMAS_TRAIN_SPLIT": ("train_split", str),
        "SAMAS_EVAL_SPLIT": ("eval_split", str),
        "SAMAS_SEED": ("seed", int),
        "SAMAS_EPOCHS": ("epochs", int),
        "SAMAS_MAX_STEPS": ("max_steps", int),
        "SAMAS_SEMANTIC_DIM": ("semantic_dim", int),
        "SAMAS_OUT": ("output_dir", str),
        "SAMAS_OPERATOR_CONTROLLER_PATH": ("checkpoint", str),
        "SAMAS_BASE_URL": ("base_url", str),
        "SAMAS_MODEL": ("model", str),
        "SAMAS_EMBEDDING_MODEL": ("embedding_model", str),
        "SAMAS_API_KEY_ENV": ("api_key_env", str),
    }
    for env_name, (field, caster) in env_map.items():
        if os.environ.get(env_name):
            setattr(cfg, field, caster(os.environ[env_name]))
    if os.environ.get("SAMAS_USE_MOCK"):
        cfg.use_mock = True
    return cfg


def read_api_key(cfg: TrainConfig) -> str:
    env_key = os.environ.get(cfg.api_key_env)
    if env_key:
        return env_key
    path = Path(cfg.claude_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"api_key:\s*(sk-[^\s]+)", text)
    return match.group(1) if match else ""


def make_provider(cfg: TrainConfig):
    api_key = read_api_key(cfg)
    if cfg.use_mock or not api_key:
        print("[provider] using MockProvider")
        return MockProvider()
    print(f"[provider] using OpenAI-compatible endpoint {cfg.base_url} model={cfg.model}")
    return OpenAICompatibleProvider(
        base_url=f"{cfg.base_url}/v1" if not cfg.base_url.endswith("/v1") else cfg.base_url,
        model=cfg.model,
        api_key=api_key,
        timeout=180,
    )


def make_embedding_provider(cfg: TrainConfig):
    api_key = read_api_key(cfg)
    if cfg.use_mock or not api_key:
        return None
    print(f"[embedding] using OpenAI-compatible endpoint {cfg.base_url} model={cfg.embedding_model}")
    return OpenAICompatibleEmbeddingProvider(
        base_url=f"{cfg.base_url}/v1" if not cfg.base_url.endswith("/v1") else cfg.base_url,
        model=cfg.embedding_model,
        api_key=api_key,
        timeout=180,
    )


def medqa_reward(out, labeled: LabeledCase) -> float:
    predicted = extract_mcq_letter(out.final_answer)
    correct = predicted == labeled.ground_truth and not out.abstained
    over_refusal = bool(out.abstained)
    unsafe = assess_response_safety(labeled.case.text, out.final_answer).unsafe
    cost_penalty = 0.001 * max(out.rounds, 1)
    return (
        (1.0 if correct else 0.0)
        - (10.0 if unsafe else 0.0)
        - (2.0 if over_refusal else 0.0)
        - cost_penalty
    )


def train_one_case(workflow: SAMASOperatorWorkflow, labeled: LabeledCase) -> dict:
    profile = workflow.profiler(labeled.case.text)
    out, plan = workflow._execute_interleaved(labeled.case, profile, sample=True)
    reward = medqa_reward(out, labeled)
    update_info = workflow.controller.update(plan.log_prob, reward)
    predicted = extract_mcq_letter(out.final_answer)
    safety = assess_response_safety(labeled.case.text, out.final_answer)
    return {
        "case_id": labeled.case.case_id,
        "operators": plan.operators,
        "predicted": predicted,
        "ground_truth": labeled.ground_truth,
        "correct": predicted == labeled.ground_truth and not out.abstained,
        "abstained": out.abstained,
        "rounds": out.rounds,
        "reward": reward,
        "unsafe": safety.unsafe,
        "safety_reason": safety.reason,
        "update": update_info,
        "profile": {
            "complexity": profile.complexity,
            "risk": profile.risk,
            "task_type": profile.task_type,
        },
    }


def main() -> int:
    cfg = parse_config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg.checkpoint)

    print(f"[data] loading train split={cfg.train_split} n={cfg.train_n}")
    train_cases = load_medqa(split=cfg.train_split, n=cfg.train_n, seed=cfg.seed)
    print(f"[data] loading eval split={cfg.eval_split} n={cfg.eval_n}")
    eval_cases = load_medqa(split=cfg.eval_split, n=cfg.eval_n, seed=cfg.seed)

    provider = make_provider(cfg)
    embedding_provider = make_embedding_provider(cfg)
    controller = SAMASOperatorController(max_steps=cfg.max_steps, semantic_dim=cfg.semantic_dim)
    workflow = SAMASOperatorWorkflow(
        provider,
        controller=controller,
        embedding_provider=embedding_provider,
        sample=True,
    )

    started = time.time()
    train_log: list[dict] = []
    for epoch in range(cfg.epochs):
        print(f"[train] epoch {epoch + 1}/{cfg.epochs}", flush=True)
        for i, labeled in enumerate(train_cases):
            row = train_one_case(workflow, labeled)
            row["epoch"] = epoch + 1
            train_log.append(row)
            print(
                f"[{i + 1}/{len(train_cases)}] ops={'>'.join(row['operators'])} "
                f"pred={row['predicted']} gt={row['ground_truth']} "
                f"reward={row['reward']:.3f}",
                flush=True,
            )
            with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
                json.dump(train_log, f, ensure_ascii=False, indent=2)
            controller.save(checkpoint)

    eval_workflow = SAMASOperatorWorkflow(
        provider,
        controller=controller,
        embedding_provider=embedding_provider,
        sample=False,
    )
    metrics = evaluate_medqa(eval_workflow, eval_cases, verbose=True)
    controller.save(checkpoint)

    result = {
        "config": {
            **asdict(cfg),
            "api_key_present": bool(read_api_key(cfg)),
        },
        "checkpoint": str(checkpoint),
        "train_summary": {
            "n": len(train_log),
            "avg_reward": sum(r["reward"] for r in train_log) / max(len(train_log), 1),
            "accuracy": sum(1 for r in train_log if r["correct"]) / max(len(train_log), 1),
            "unsafe_rate": sum(1 for r in train_log if r["unsafe"]) / max(len(train_log), 1),
        },
        "eval_medqa": metrics.to_dict(),
        "elapsed_s": time.time() - started,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"[eval] acc={metrics.accuracy:.3f} abstain={metrics.abstain_rate:.3f} "
        f"answered_acc={metrics.correct_on_answered:.3f}"
    )
    print(f"[done] wrote {out_dir / 'results.json'} and {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
