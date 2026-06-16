from __future__ import annotations

from ..agents import Agent
from ..models import AgentResponse, LLMMessage, PatientCase
from ..providers import LLMProvider


ACCURACY_PROMPT = """You are a careful medical benchmark solver.
If the input is multiple-choice, choose exactly one option and end with:
Final answer: <A/B/C/D/E>

If the input is not multiple-choice, give a concise safe medical information response.
Do not provide dangerous operational medical instructions.
"""


class AccuracyPath:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.agent = Agent("AccuracyPath", ACCURACY_PROMPT, provider)

    def run(self, case: PatientCase) -> tuple[str, list[AgentResponse], int]:
        response = self.agent.run(case.text, temperature=0.0)
        return response.content, [response], 1
