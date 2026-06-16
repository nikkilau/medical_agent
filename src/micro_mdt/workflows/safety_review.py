from __future__ import annotations

from ..agents import Agent, parse_verdict
from ..models import AgentResponse, DebateRound, PatientCase, Verdict
from ..prompts import PHARMACY_PROMPT, REVISION_PROMPT, SAFETY_PROMPT
from ..providers import LLMProvider


SAFE_DRAFT_PROMPT = """You are a medical safety response agent.
For harmful or unsafe medical requests, refuse to provide operational instructions.
Give a brief reason and redirect to safe, professional, or emergency help when appropriate.
Do not include dosing, procedural steps, or ways to bypass clinicians.
End with: Final answer: ABSTAIN
"""


REFUSAL_TEXT = (
    "I cannot help provide harmful or unsafe medical instructions. "
    "Please seek help from a qualified clinician or emergency services if there is immediate risk. "
    "Final answer: ABSTAIN"
)


class SafetyReviewPath:
    def __init__(self, provider: LLMProvider, *, max_rounds: int) -> None:
        self.provider = provider
        self.max_rounds = max_rounds
        self.safe_draft = Agent("SafetyDraft", SAFE_DRAFT_PROMPT, provider)
        self.pharmacist = Agent("Pharmacist", PHARMACY_PROMPT, provider)
        self.safety = Agent("SafetyEthics", SAFETY_PROMPT, provider)
        self.reviser = Agent("SafetyReviser", REVISION_PROMPT, provider)

    def run(self, case: PatientCase) -> tuple[str, list[DebateRound], list[AgentResponse], bool, int]:
        proposal = self.safe_draft.run(case.text, temperature=0.0)
        trace: list[AgentResponse] = [proposal]
        rounds: list[DebateRound] = []

        for round_index in range(1, self.max_rounds + 1):
            pharmacy = self.pharmacist.run(self._review_input(case, proposal.content), temperature=0.0)
            safety = self.safety.run(self._review_input(case, proposal.content), temperature=0.0)
            pharmacy.verdict = parse_verdict(pharmacy.content)
            safety.verdict = parse_verdict(safety.content)
            rounds.append(DebateRound(round_index, proposal, pharmacy, safety))
            trace.extend([pharmacy, safety])

            if pharmacy.verdict is Verdict.PASS and safety.verdict is Verdict.PASS:
                return proposal.content, rounds, trace, False, round_index
            if pharmacy.verdict is Verdict.ABSTAIN or safety.verdict is Verdict.ABSTAIN:
                return REFUSAL_TEXT, rounds, trace, True, round_index
            if round_index < self.max_rounds:
                feedback = (
                    f"Pharmacy: {pharmacy.content}\n"
                    f"SafetyEthics: {safety.content}\n"
                    "Revise into a safe refusal or safe redirection if the request is harmful."
                )
                proposal = self.reviser.run(
                    f"病例/请求：\n{case.text}\n\n上一版回应：\n{proposal.content}\n\n审查意见：\n{feedback}",
                    temperature=0.0,
                )
                trace.append(proposal)

        return REFUSAL_TEXT, rounds, trace, True, self.max_rounds

    @staticmethod
    def _review_input(case: PatientCase, proposal: str) -> str:
        return f"病例：\n{case.text}\n\n待审查方案：\n{proposal}"
