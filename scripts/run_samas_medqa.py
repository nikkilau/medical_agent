"""Run the SAMAS prompt-profiler pilot on MedQA only.

Examples:
    SAMAS_USE_MOCK=1 SAMAS_N_MEDQA=5 PYTHONPATH=src python scripts/run_samas_medqa.py

    GPUSTACK_API_KEY=... SAMAS_N_MEDQA=20 PYTHONPATH=src python scripts/run_samas_medqa.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medical_ttc.data import load_medqa  # noqa: E402
from medical_ttc.evaluator import evaluate_medqa  # noqa: E402
from micro_mdt.providers import MockProvider, OpenAICompatibleProvider  # noqa: E402
from samas.workflow import SAMASPromptWorkflow  # noqa: E402


@dataclass
class SAMASMedQAConfig:
    n_medqa: int = 10
    seed: int = 42
    split: str = "test"
    output_dir: str = "reports/samas_medqa_pilot"
    use_mock: bool = False
    base_url: str = "http://202.120.24.199:13000/v1"
    model: str = "qwen2.5:14b"
    api_key_env: str = "GPUSTACK_API_KEY"
    enable_review: bool = False


def parse_config() -> SAMASMedQAConfig:
    cfg = SAMASMedQAConfig()
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
    if os.environ.get("SAMAS_API_KEY_ENV"):
        cfg.api_key_env = os.environ["SAMAS_API_KEY_ENV"]
    if os.environ.get("SAMAS_ENABLE_REVIEW"):
        cfg.enable_review = True
    if os.environ.get("SAMAS_USE_MOCK"):
        cfg.use_mock = True
    if not os.environ.get(cfg.api_key_env):
        cfg.use_mock = True
    return cfg


def make_provider(cfg: SAMASMedQAConfig):
    if cfg.use_mock:
        print("[provider] using MockProvider (no API key found or SAMAS_USE_MOCK=1)")
        return MockProvider()
    return OpenAICompatibleProvider(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=os.environ[cfg.api_key_env],
        timeout=120,
    )


def main() -> int:
    cfg = parse_config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading MedQA split={cfg.split} n={cfg.n_medqa} seed={cfg.seed}")
    cases = load_medqa(split=cfg.split, n=cfg.n_medqa, seed=cfg.seed)
    print(f"[data] got {len(cases)} cases")

    provider = make_provider(cfg)
    workflow = SAMASPromptWorkflow(provider, enable_review=cfg.enable_review)

    started = time.time()
    metrics = evaluate_medqa(workflow, cases, verbose=True)
    elapsed = time.time() - started

    record = {
        "config": asdict(cfg),
        "started": started,
        "elapsed_s": elapsed,
        "workflow": workflow.name,
        "medqa": metrics.to_dict(),
    }
    result_path = out_dir / "results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(
        "[done] "
        f"acc={metrics.accuracy:.3f} "
        f"abstain={metrics.abstain_rate:.3f} "
        f"answered_acc={metrics.correct_on_answered:.3f} "
        f"avg_rounds={metrics.avg_rounds:.2f} "
        f"avg_elapsed={metrics.avg_elapsed_s:.2f}s "
        f"wall={elapsed:.1f}s"
    )
    print(f"[done] wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
