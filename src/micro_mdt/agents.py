from __future__ import annotations

from .models import AgentResponse, Difficulty, LLMMessage, Verdict
from .prompts import (
    DECISION_SYNTHESIS_PROMPT,
    DOCUMENTATION_PROMPT,
    GENERALIST_PROMPT,
    PHARMACY_PROMPT,
    REVISION_PROMPT,
    SAFETY_PROMPT,
    TRIAGE_PROMPT,
)
from .providers import LLMProvider


def parse_difficulty(text: str) -> Difficulty:
    upper = text.upper()
    if "[HIGH]" in upper:
        return Difficulty.HIGH
    if "[MEDIUM]" in upper:
        return Difficulty.MEDIUM
    return Difficulty.LOW


def parse_verdict(text: str) -> Verdict:
    upper = text.upper()
    if "[ABSTAIN]" in upper:
        return Verdict.ABSTAIN
    if "[PASS]" in upper:
        return Verdict.PASS
    return Verdict.REVISE


class Agent:
    def __init__(self, name: str, system_prompt: str, provider: LLMProvider) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider

    def run(self, user_content: str, *, temperature: float = 0.2) -> AgentResponse:
        content = self.provider.complete(
            [
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=user_content),
            ],
            temperature=temperature,
        )
        return AgentResponse(agent=self.name, content=content)


class AgentSuite:
    def __init__(self, provider: LLMProvider) -> None:
        self.triage = Agent("Triage", TRIAGE_PROMPT, provider)
        self.generalist = Agent("Generalist", GENERALIST_PROMPT, provider)
        self.reviser = Agent("GeneralistRevision", REVISION_PROMPT, provider)
        self.pharmacist = Agent("Pharmacist", PHARMACY_PROMPT, provider)
        self.safety = Agent("SafetyEthics", SAFETY_PROMPT, provider)
        self.decision_synth = Agent("DecisionSynthesis", DECISION_SYNTHESIS_PROMPT, provider)
        self.documentation = Agent("Documentation", DOCUMENTATION_PROMPT, provider)

