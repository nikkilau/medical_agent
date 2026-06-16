"""W4b — Medical TTC with the LLM-based ZeroShotRouter (instead of the
keyword HeuristicRouter used by W4 default).

The pilot showed W4-default failed safety (unsafe=0.7) because the keyword
router missed adversarial MedSafetyBench prompts and routed them to LOW
tier (bypassing the safety骨架). This run validates the hypothesis that
the LLM-based Triage agent — the same one W2 uses — correctly routes
those prompts to HIGH so the safety review fires.

Env vars:
    HF_ENDPOINT        default https://hf-mirror.com
    MEDTTC_N_MEDQA     default 20
    MEDTTC_N_SAFETY    default 10
    MEDTTC_OUT         default reports/w4b
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
from medical_ttc.router import ZeroShotRouter  # noqa: E402
from medical_ttc.ttc_workflow import MedicalTTCWorkflow  # noqa: E402
from micro_mdt.providers import OpenAICompatibleProvider  # noqa: E402


def main() -> int:
    out_dir = Path(os.environ.get("MEDTTC_OUT", "reports/w4b"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_medqa = int(os.environ.get("MEDTTC_N_MEDQA", "20"))
    n_safety = int(os.environ.get("MEDTTC_N_SAFETY", "10"))

    medqa = load_medqa(split="test", n=n_medqa, seed=42)
    medsafety = load_medsafety(n=n_safety, seed=42)

    executor = OpenAICompatibleProvider(
        base_url="http://202.120.5.12:18080/v1",
        model="qwen2.5:14b",
        api_key="gpustack_4ba10aa3dd610a36_843f703126bc1aa7b9c05c7486a388be",
    )

    print(f"=== W4b — Medical TTC + ZeroShotRouter (LLM Triage) ===")
    wf = MedicalTTCWorkflow(
        executor, router=ZeroShotRouter(executor), name="W4b_MedicalTTC_LLMRouter"
    )
    t0 = time.time()
    m = evaluate_medqa(wf, medqa, verbose=True)
    s = evaluate_medsafety(wf, medsafety, verbose=True)
    g = compute_g_score(m, s)
    rec = {
        "W4b_MedicalTTC_LLMRouter": {
            "medqa": m.to_dict(),
            "medsafety": s,
            "g_score": g,
            "wall_s_medqa": time.time() - t0,
        }
    }
    (out_dir / "results.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"  acc={m.accuracy:.3f} abstain={m.abstain_rate:.3f} "
          f"unsafe={s['unsafe_rate']:.3f} G={g:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
