"""W7 — DSPy-style prompt-only optimization baseline.

We do NOT take a hard dependency on `dspy-ai` (the SJTU server cannot reach
the DSPy installation source). Instead we approximate the spirit of
DSPy/MIPRO with a small prompt grid-search:

    * Fix the W2 structure (Hand-coded Micro-MDT skeleton).
    * Generate `n_candidates` alternative system prompts for the Generalist
      operator (via the optimizer LLM).
    * Score each candidate on a small validation slice.
    * Return the workflow whose Generalist prompt scored best.

This is a deliberate simplification — it lets us answer the AFLOW vs
prompt-only question without coupling to DSPy infrastructure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from micro_mdt.models import LLMMessage, PatientCase
from micro_mdt.providers import LLMProvider
from .baselines import HandCodedMicroMDTWorkflow, extract_mcq_letter
from .data import LabeledCase
from .evaluator import evaluate_medqa
from .workflow import Workflow, WorkflowResult


CANDIDATE_PROMPT_SYSTEM = """You are a prompt engineer. Generate ONE alternative
system-prompt for a medical generalist agent answering multiple-choice
USMLE-style questions. The agent must produce a single A/B/C/D letter as
its final answer.

Return ONLY the system-prompt text, no explanations, no preamble."""


CANDIDATE_PROMPT_USER = """The current system-prompt is:

---
{current}
---

Generate ONE alternative system-prompt that may improve accuracy. It must:
- Be in English or Chinese (whichever fits the audience),
- Tell the model to think briefly then answer with a single letter,
- Avoid removing safety guard rails about medical advice."""


@dataclass
class PromptGridResult:
    best_prompt: str
    candidates: list[str]
    scores: list[float]


def generate_candidate_prompts(
    optimizer: LLMProvider,
    current_prompt: str,
    *,
    n_candidates: int = 4,
    temperature: float = 0.8,
) -> list[str]:
    out = []
    for _ in range(n_candidates):
        try:
            text = optimizer.complete(
                [
                    LLMMessage(role="system", content=CANDIDATE_PROMPT_SYSTEM),
                    LLMMessage(
                        role="user",
                        content=CANDIDATE_PROMPT_USER.format(current=current_prompt[:600]),
                    ),
                ],
                temperature=temperature,
            )
            text = text.strip()
            if text and len(text) < 1500:
                out.append(text)
        except Exception:
            continue
    return out


def run_prompt_grid_search(
    executor: LLMProvider,
    optimizer: LLMProvider,
    validation_cases: list[LabeledCase],
    *,
    initial_prompt: str | None = None,
    n_candidates: int = 4,
) -> PromptGridResult:
    """Search over n_candidates alternative generalist prompts."""
    from micro_mdt.prompts import GENERALIST_PROMPT

    initial = initial_prompt or GENERALIST_PROMPT
    candidates = [initial] + generate_candidate_prompts(
        optimizer, initial, n_candidates=n_candidates
    )

    scores: list[float] = []
    for i, prompt in enumerate(candidates):
        wf = _build_w2_with_prompt(executor, prompt)
        m = evaluate_medqa(wf, validation_cases)
        scores.append(m.accuracy)
        print(f"[W7 grid] candidate {i}: acc={m.accuracy:.3f} (len={len(prompt)})")

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return PromptGridResult(
        best_prompt=candidates[best_idx], candidates=candidates, scores=scores
    )


def _build_w2_with_prompt(provider: LLMProvider, generalist_prompt: str) -> Workflow:
    """Build a Hand-Coded Micro-MDT but with a custom generalist prompt."""
    # We monkey-patch the prompts module copy — minimal-surface approach so
    # we don't have to add a "prompt override" parameter to MicroMDT.
    from micro_mdt.workflow import MicroMDT
    from micro_mdt.agents import AgentSuite, Agent
    from micro_mdt.prompts import (
        PHARMACY_PROMPT, SAFETY_PROMPT, REVISION_PROMPT,
        DECISION_SYNTHESIS_PROMPT, DOCUMENTATION_PROMPT, TRIAGE_PROMPT,
    )

    class CustomAgentSuite(AgentSuite):
        def __init__(self, p):
            self.triage = Agent("Triage", TRIAGE_PROMPT, p)
            self.generalist = Agent("Generalist", generalist_prompt, p)
            self.reviser = Agent("GeneralistRevision", REVISION_PROMPT, p)
            self.pharmacist = Agent("Pharmacist", PHARMACY_PROMPT, p)
            self.safety = Agent("SafetyEthics", SAFETY_PROMPT, p)
            self.decision_synth = Agent("DecisionSynthesis", DECISION_SYNTHESIS_PROMPT, p)
            self.documentation = Agent("Documentation", DOCUMENTATION_PROMPT, p)

    eng = MicroMDT(provider)
    eng.agents = CustomAgentSuite(provider)

    class _Wrapped(Workflow):
        def __init__(self):
            super().__init__("W7_DSPy_search")
            self.eng = eng

        def __call__(self, case: PatientCase) -> WorkflowResult:
            out = self.eng.run_case(case)
            return WorkflowResult(
                case=case,
                final_answer=out.final_answer,
                abstained=bool(out.human_required),
                rounds=len(out.rounds) if out.rounds else 1,
                difficulty=out.difficulty.value if hasattr(out.difficulty, "value") else str(out.difficulty),
                trace=[{"agent": r.agent, "content": r.content[:200]} for r in (out.trace or [])],
            )

    return _Wrapped()


class DSPyPromptOnlyWorkflow(Workflow):
    """The final W7 workflow — wraps the best-found prompt."""

    def __init__(self, executor: LLMProvider, best_prompt: str) -> None:
        super().__init__("W7_DSPy_PromptOnly")
        self.inner = _build_w2_with_prompt(executor, best_prompt)

    def __call__(self, case: PatientCase) -> WorkflowResult:
        return self.inner(case)
