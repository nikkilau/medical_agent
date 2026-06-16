"""Baseline workflows W0..W2.

W0: BaselineIO        — single LLM call, no TTC
W1: CoTSCWorkflow     — fixed N-sample self-consistency (Snell-style pure TTC)
W2: HandCodedMicroMDT — current hand-written Micro-MDT (compute-tier adaptive)
"""

from __future__ import annotations

from collections import Counter
import re

from micro_mdt.models import PatientCase
from micro_mdt.providers import LLMProvider
from .operators import (
    EnsembleOperator,
    GenerateOperator,
)
from .workflow import Workflow, WorkflowResult


# ----------------------------- helpers ---------------------------------

MCQ_LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def extract_mcq_letter(text: str) -> str | None:
    """Best-effort extraction of a single A-E letter answer.

    Looks for patterns like 'Answer: B', 'final answer is C', '(D)', etc.
    Falls back to last A-E letter token in the text.
    """
    if not text:
        return None
    patterns = [
        r"(?:final\s*answer|答案)[^A-E\n]*([A-E])",
        r"answer\s*[:：]\s*([A-E])",
        r"\b([A-E])\s*[)\.]\s",
        r"^\s*([A-E])\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    letters = MCQ_LETTER_RE.findall(text)
    if letters:
        return letters[-1].upper()
    return None


# ----------------------------- W0 ---------------------------------

class BaselineIOWorkflow(Workflow):
    """W0 — single LLM call, no TTC."""

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__("W0_BaselineIO")
        self.generate = GenerateOperator(provider)

    def __call__(self, case: PatientCase) -> WorkflowResult:
        out = self.generate(case.text, temperature=0.0)
        return WorkflowResult(
            case=case,
            final_answer=out,
            rounds=1,
            difficulty="N/A",
            trace=[{"step": "generate", "output": out[:200]}],
        )


# ----------------------------- W1 ---------------------------------

class CoTSCWorkflow(Workflow):
    """W1 — Snell-style pure TTC: N-sample self-consistency, no difficulty routing."""

    def __init__(self, provider: LLMProvider, n_samples: int = 5) -> None:
        super().__init__(f"W1_CoTSC_n{n_samples}")
        self.generate = GenerateOperator(provider)
        self.ensemble = EnsembleOperator(provider, n_samples=n_samples)
        self.n_samples = n_samples

    def __call__(self, case: PatientCase) -> WorkflowResult:
        final, samples = self.ensemble(
            self.generate, case.text, extract_answer=extract_mcq_letter
        )
        return WorkflowResult(
            case=case,
            final_answer=final,
            rounds=self.n_samples,
            difficulty="N/A",
            trace=[{"step": "ensemble", "n_samples": self.n_samples, "winner": final[:200]}],
            metadata={"all_samples": samples},
        )


# ----------------------------- W2 ---------------------------------

class HandCodedMicroMDTWorkflow(Workflow):
    """W2 — wraps the existing hand-written MicroMDT logic.

    Delegates to the existing workflow.MicroMDT so behaviour is
    identical to the pre-refactor implementation; this lets us evaluate
    the original system on the same benchmark harness.
    """

    def __init__(self, provider: LLMProvider, max_rounds: int = 3) -> None:
        super().__init__(f"W2_HandCoded_r{max_rounds}")
        from micro_mdt.workflow import MicroMDT  # lazy import to avoid cycles

        self.engine = MicroMDT(provider, max_rounds=max_rounds)

    def __call__(self, case: PatientCase) -> WorkflowResult:
        mdt_result = self.engine.run_case(case)
        abstained = bool(mdt_result.human_required)
        rounds = len(mdt_result.rounds) if mdt_result.rounds else 1
        return WorkflowResult(
            case=case,
            final_answer=mdt_result.final_answer,
            abstained=abstained,
            rounds=rounds,
            difficulty=mdt_result.difficulty.value
            if hasattr(mdt_result.difficulty, "value")
            else str(mdt_result.difficulty),
            trace=[
                {"agent": r.agent, "content": r.content[:200]}
                for r in (mdt_result.trace or [])
            ],
            metadata={"status": mdt_result.status},
        )

