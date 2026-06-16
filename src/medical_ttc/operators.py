"""Reusable Operator components.

Each Operator wraps an existing Agent and exposes a clean
callable interface. AFLOW MCTS searches over compositions of these
operators (plus prompts).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from micro_mdt.agents import Agent, parse_difficulty, parse_verdict
from micro_mdt.models import Difficulty, Verdict
from micro_mdt.prompts import (
    DECISION_SYNTHESIS_PROMPT,
    GENERALIST_PROMPT,
    PHARMACY_PROMPT,
    REVISION_PROMPT,
    SAFETY_PROMPT,
    TRIAGE_PROMPT,
)
from micro_mdt.providers import LLMProvider


class Operator:
    """Base callable component used inside workflows."""

    def __init__(self, name: str, provider: LLMProvider) -> None:
        self.name = name
        self.provider = provider


class TriageOperator(Operator):
    """Classify a case into LOW/MEDIUM/HIGH compute tier."""

    def __init__(self, provider: LLMProvider, prompt: str | None = None) -> None:
        super().__init__("Triage", provider)
        self.agent = Agent("Triage", prompt or TRIAGE_PROMPT, provider)

    def __call__(self, case_text: str) -> tuple[Difficulty, str]:
        resp = self.agent.run(case_text)
        return parse_difficulty(resp.content), resp.content


class GenerateOperator(Operator):
    """Generate an initial proposal (generalist)."""

    def __init__(self, provider: LLMProvider, prompt: str | None = None) -> None:
        super().__init__("Generate", provider)
        self.agent = Agent("Generalist", prompt or GENERALIST_PROMPT, provider)

    def __call__(self, case_text: str, *, temperature: float = 0.2) -> str:
        return self.agent.run(case_text, temperature=temperature).content


class ReviewOperator(Operator):
    """Run pharmacy + safety reviewers in parallel-conceptually, return verdicts."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        pharmacy_prompt: str | None = None,
        safety_prompt: str | None = None,
    ) -> None:
        super().__init__("Review", provider)
        self.pharmacy = Agent("Pharmacist", pharmacy_prompt or PHARMACY_PROMPT, provider)
        self.safety = Agent("SafetyEthics", safety_prompt or SAFETY_PROMPT, provider)

    def __call__(self, case_text: str, proposal: str) -> dict:
        review_input = f"病例：\n{case_text}\n\n待审查方案：\n{proposal}"
        pharm = self.pharmacy.run(review_input)
        safe = self.safety.run(review_input)
        pharm_v = parse_verdict(pharm.content)
        safe_v = parse_verdict(safe.content)
        return {
            "pharmacy_verdict": pharm_v,
            "pharmacy_content": pharm.content,
            "safety_verdict": safe_v,
            "safety_content": safe.content,
            "pass": pharm_v is Verdict.PASS and safe_v is Verdict.PASS,
            "abstain": pharm_v is Verdict.ABSTAIN or safe_v is Verdict.ABSTAIN,
        }


class ReviseOperator(Operator):
    """Revise a proposal given reviewer feedback."""

    def __init__(self, provider: LLMProvider, prompt: str | None = None) -> None:
        super().__init__("Revise", provider)
        self.agent = Agent("Reviser", prompt or REVISION_PROMPT, provider)

    def __call__(self, case_text: str, prior_proposal: str, feedback: str) -> str:
        prompt = (
            f"病例：\n{case_text}\n\n"
            f"上一版方案：\n{prior_proposal}\n\n"
            f"审查意见：\n{feedback}"
        )
        return self.agent.run(prompt).content


class EnsembleOperator(Operator):
    """Self-consistency ensemble.

    For multiple-choice tasks (like MedQA), generate N proposals at higher
    temperature, then return the most-voted final letter.
    """

    def __init__(self, provider: LLMProvider, n_samples: int = 5) -> None:
        super().__init__("Ensemble", provider)
        self.n_samples = n_samples

    def __call__(
        self,
        generate_op: GenerateOperator,
        case_text: str,
        *,
        extract_answer: callable | None = None,
        temperature: float = 0.7,
    ) -> tuple[str, list[str]]:
        samples = [
            generate_op(case_text, temperature=temperature)
            for _ in range(self.n_samples)
        ]
        if extract_answer is None:
            # Fall back to majority over full strings
            counter = Counter(samples)
            winner = counter.most_common(1)[0][0]
            return winner, samples
        answers = [extract_answer(s) for s in samples]
        non_null = [a for a in answers if a]
        if not non_null:
            return samples[0], samples
        winning_answer = Counter(non_null).most_common(1)[0][0]
        # Return the sample that voted for the winner (first one)
        for s, a in zip(samples, answers):
            if a == winning_answer:
                return s, samples
        return samples[0], samples


class ModeratorOperator(Operator):
    """MDAgents-style moderator: synthesize multiple agents' opinions."""

    def __init__(self, provider: LLMProvider, prompt: str | None = None) -> None:
        super().__init__("Moderator", provider)
        self.agent = Agent("Moderator", prompt or DECISION_SYNTHESIS_PROMPT, provider)

    def __call__(self, case_text: str, views: Iterable[str]) -> str:
        joined = "\n\n".join(f"专家 {i + 1}:\n{v}" for i, v in enumerate(views))
        return self.agent.run(f"病例：\n{case_text}\n\n各专家意见：\n{joined}").content


class CustomOperator(Operator):
    """Free-form node — AFLOW MCTS mutates this operator's system prompt."""

    def __init__(
        self, provider: LLMProvider, system_prompt: str, name: str = "Custom"
    ) -> None:
        super().__init__(name, provider)
        self.agent = Agent(name, system_prompt, provider)

    def __call__(self, user_input: str, *, temperature: float = 0.2) -> str:
        return self.agent.run(user_input, temperature=temperature).content
