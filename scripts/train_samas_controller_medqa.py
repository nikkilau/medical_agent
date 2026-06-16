"""Train the SAMAS MaAS-style template controller on MedQA.

This is the first runnable controller version:
PromptProfiler-ZS -> SAMASTemplateController -> execute one full template
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
from micro_mdt.providers import MockProvider, OpenAICompatibleProvider  # noqa: E402
from samas import SAMASControllerWorkflow, SAMASTemplateController  # noqa: E402


@dataclass
class TrainConfig:
    train_n: int = 20
    eval_n: int = 20
    train_split: str = "train"
    eval_split: str = "test"
    seed: int = 42
    epochs: int = 1
    output_dir: str = "reports/samas_controller_medqa"
    checkpoint: str = "reports/samas_controller_medqa/controller.pt"
    use_mock: bool = False
    base_url: str = "http://202.120.24.199:13000/v1"
    model: str = "qwen2.5:14b"
    api_key_env: str = "GPUSTACK_API_KEY"
    claude_path: str = "/home/sjtu/workspace/ycy/CLAUDE.md"
    enable_review: bool = True


def parse_config() -> TrainConfig:
    cfg = TrainConfig()
    env_map = {
        "SAMAS_TRAIN_N": ("train_n", int),
        "SAMAS_EVAL_N": ("eval_n", int),
        "SAMAS_TRAIN_SPLIT": ("train_split", str),
        "SAMAS_EVAL_SPLIT": ("eval_split", str),
        "SAMAS_SEED": ("seed", int),
        "SAMAS_EPOCHS": ("epochs", int),
        "SAMAS_OUT": ("output_dir", str),
        "SAMAS_CONTROLLER_PATH": ("checkpoint", str),
        "SAMAS_BASE_URL": ("base_url", str),
        "SAMAS_MODEL": ("model", str),
        "SAMAS_API_KEY_ENV": ("api_key_env", str),
    }
    for env_name, (field, caster) in env_map.items():
        if os.environ.get(env_name):
            setattr(cfg, field, caster(os.environ[env_name]))
    if os.environ.get("SAMAS_USE_MOCK"):
        cfg.use_mock = True
    if os.environ.get("SAMAS_DISABLE_REVIEW"):
        cfg.enable_review = False
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


def medqa_reward(out, labeled: LabeledCase) -> float:
    predicted = extract_mcq_letter(out.final_answer)
    correct = predicted == labeled.ground_truth and not out.abstained
    over_refusal = bool(out.abstained)
    cost_penalty = 0.001 * max(out.rounds, 1)
    return (1.0 if correct else 0.0) - (2.0 if over_refusal else 0.0) - cost_penalty


def train_one_case(workflow: SAMASControllerWorkflow, labeled: LabeledCase) -> dict:
    profile = workflow.profiler(labeled.case.text)
    decision = workflow.controller.select_template(labeled.case.text, profile, sample=True)
    out = workflow._execute_decision(
        labeled.case,
        profile,
        decision,
        controller_metadata={
            "controller": {
                "type": "SAMASTemplateController",
                "sample": True,
                "allowed_templates": list(decision.allowed_templates),
                "probs": decision.probs,
            }
        },
    )
    reward = medqa_reward(out, labeled)
    update_info = workflow.controller.update(decision.log_prob, reward)
    predicted = extract_mcq_letter(out.final_answer)
    return {
        "case_id": labeled.case.case_id,
        "template": decision.template_id,
        "predicted": predicted,
        "ground_truth": labeled.ground_truth,
        "correct": predicted == labeled.ground_truth and not out.abstained,
        "abstained": out.abstained,
        "rounds": out.rounds,
        "reward": reward,
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
    controller = SAMASTemplateController()
    workflow = SAMASControllerWorkflow(
        provider,
        controller=controller,
        sample=True,
        enable_review=cfg.enable_review,
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
                f"[{i + 1}/{len(train_cases)}] template={row['template']} "
                f"pred={row['predicted']} gt={row['ground_truth']} "
                f"reward={row['reward']:.3f}",
                flush=True,
            )
            with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
                json.dump(train_log, f, ensure_ascii=False, indent=2)
            controller.save(checkpoint)

    eval_workflow = SAMASControllerWorkflow(
        provider,
        controller=controller,
        sample=False,
        enable_review=cfg.enable_review,
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
