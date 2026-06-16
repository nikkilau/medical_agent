from __future__ import annotations

import re
from collections.abc import Callable

from .agents import AgentSuite, parse_difficulty, parse_verdict
from .models import AgentResponse, DebateRound, Difficulty, MDTResult, PatientCase, Verdict
from .prompts import DISCLAIMER
from .providers import LLMProvider

HumanDecisionCallback = Callable[[MDTResult], "str | None"]


class MicroMDT:
    def __init__(self, provider: LLMProvider, *, max_rounds: int = 3) -> None:
        self.agents = AgentSuite(provider)
        self.max_rounds = max_rounds

    def run_case(
        self,
        patient_case: PatientCase,
        *,
        human_decision_callback: HumanDecisionCallback | None = None,
    ) -> MDTResult:
        triage = self.agents.triage.run(patient_case.text)
        difficulty = parse_difficulty(triage.content)

        if difficulty is Difficulty.LOW:
            proposal = self.agents.generalist.run(patient_case.text)
            return MDTResult(
                case=patient_case,
                difficulty=difficulty,
                status="completed_low_compute",
                final_answer=self._with_disclaimer(proposal.content),
                trace=[triage, proposal],
            )

        if difficulty is Difficulty.MEDIUM:
            result = self._run_medium(patient_case, triage)
        else:
            result = self._run_high(patient_case, triage)

        if result.human_required:
            result.human_options = self._synthesize_options(result)
            if human_decision_callback:
                decision = human_decision_callback(result)
                if decision:
                    result.human_decision = decision
                    result.documents = self._generate_documents(result, decision)
                    result.status = "human_decision_documented"
        return result

    def _run_medium(self, patient_case: PatientCase, triage: AgentResponse) -> MDTResult:
        proposal = self.agents.generalist.run(patient_case.text)
        pharmacy = self.agents.pharmacist.run(self._review_input(patient_case, proposal.content))
        safety = self.agents.safety.run(self._review_input(patient_case, proposal.content))
        pharmacy.verdict = parse_verdict(pharmacy.content)
        safety.verdict = parse_verdict(safety.content)
        round1 = DebateRound(1, proposal, pharmacy, safety)
        rounds = [round1]
        trace = [triage, proposal, pharmacy, safety]

        if pharmacy.verdict is Verdict.PASS and safety.verdict is Verdict.PASS:
            return MDTResult(
                case=patient_case,
                difficulty=Difficulty.MEDIUM,
                status="completed_medium_reviewed",
                final_answer=self._with_disclaimer("已完成一次交叉复核。\n\n" + proposal.content),
                rounds=rounds,
                trace=trace,
            )

        if pharmacy.verdict is Verdict.ABSTAIN or safety.verdict is Verdict.ABSTAIN:
            return MDTResult(
                case=patient_case,
                difficulty=Difficulty.MEDIUM,
                status="needs_human_review",
                final_answer=self._with_disclaimer(self._abstention_message(rounds)),
                rounds=rounds,
                human_required=True,
                trace=trace,
            )

        feedback = self._feedback(round1)
        revised = self.agents.reviser.run(
            f"病例：\n{patient_case.text}\n\n上一版方案：\n{proposal.content}\n\n审查意见：\n{feedback}"
        )
        pharmacy2 = self.agents.pharmacist.run(self._review_input(patient_case, revised.content))
        safety2 = self.agents.safety.run(self._review_input(patient_case, revised.content))
        pharmacy2.verdict = parse_verdict(pharmacy2.content)
        safety2.verdict = parse_verdict(safety2.content)
        round2 = DebateRound(2, revised, pharmacy2, safety2)
        rounds.append(round2)
        trace.extend([revised, pharmacy2, safety2])

        if pharmacy2.verdict is Verdict.PASS and safety2.verdict is Verdict.PASS:
            return MDTResult(
                case=patient_case,
                difficulty=Difficulty.MEDIUM,
                status="completed_medium_revised",
                final_answer=self._with_disclaimer("经过一轮修订和复核，MDT 达成共识。\n\n" + revised.content),
                rounds=rounds,
                trace=trace,
            )

        return MDTResult(
            case=patient_case,
            difficulty=Difficulty.MEDIUM,
            status="needs_human_review",
            final_answer=self._with_disclaimer(self._abstention_message(rounds)),
            rounds=rounds,
            human_required=True,
            trace=trace,
        )

    def _run_high(self, patient_case: PatientCase, triage: AgentResponse) -> MDTResult:
        proposal = self.agents.generalist.run(patient_case.text)
        rounds: list[DebateRound] = []
        trace = [triage, proposal]

        for index in range(1, self.max_rounds + 1):
            pharmacy = self.agents.pharmacist.run(self._review_input(patient_case, proposal.content))
            safety = self.agents.safety.run(self._review_input(patient_case, proposal.content))
            pharmacy.verdict = parse_verdict(pharmacy.content)
            safety.verdict = parse_verdict(safety.content)
            round_record = DebateRound(index, proposal, pharmacy, safety)
            rounds.append(round_record)
            trace.extend([pharmacy, safety])

            if pharmacy.verdict is Verdict.PASS and safety.verdict is Verdict.PASS:
                return MDTResult(
                    case=patient_case,
                    difficulty=Difficulty.HIGH,
                    status="consensus_reached",
                    final_answer=self._with_disclaimer("MDT 达成共识。\n\n" + proposal.content),
                    rounds=rounds,
                    trace=trace,
                )

            if pharmacy.verdict is Verdict.ABSTAIN or safety.verdict is Verdict.ABSTAIN:
                break

            feedback = self._feedback(round_record)
            proposal = self.agents.reviser.run(
                f"病例：\n{patient_case.text}\n\n上一版方案：\n{proposal.content}\n\n审查意见：\n{feedback}"
            )
            trace.append(proposal)

        return MDTResult(
            case=patient_case,
            difficulty=Difficulty.HIGH,
            status="abstained_needs_human",
            final_answer=self._with_disclaimer(self._abstention_message(rounds)),
            rounds=rounds,
            human_required=True,
            trace=trace,
        )

    def _synthesize_options(self, result: MDTResult) -> str:
        round_summary = "\n\n".join(self._feedback(r) for r in result.rounds)
        response = self.agents.decision_synth.run(
            f"病例：\n{result.case.text}\n\nAI 分歧摘要：\n{round_summary}"
        )
        return response.content

    def _generate_documents(self, result: MDTResult, decision: str) -> dict[str, str]:
        round_summary = "\n\n".join(self._feedback(r) for r in result.rounds)
        response = self.agents.documentation.run(
            f"病例：\n{result.case.text}\n\nAI 分歧摘要：\n{round_summary}\n\n医生最终决策：\n{decision}"
        )
        return self._parse_documents(response.content)

    @staticmethod
    def _parse_documents(text: str) -> dict[str, str]:
        docs: dict[str, str] = {}
        sections = {
            "FOLLOWUP": "followup_draft",
            "PATIENT_NOTE": "patient_note",
            "REMINDER": "reminder",
        }
        for marker, key in sections.items():
            pattern = rf"\[{marker}\]\s*(.*?)(?=\[(?:{'|'.join(sections)})\]|\Z)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                docs[key] = match.group(1).strip()
        return docs if docs else {"doctor_draft": text}

    @staticmethod
    def _review_input(patient_case: PatientCase, proposal: str) -> str:
        return f"病例：\n{patient_case.text}\n\n待审查方案：\n{proposal}"

    @staticmethod
    def _feedback(round_record: DebateRound) -> str:
        return (
            f"第 {round_record.round_index} 轮审查：\n"
            f"药师：{round_record.pharmacy_review.content}\n"
            f"安全伦理：{round_record.safety_review.content}"
        )

    def _abstention_message(self, rounds: list[DebateRound]) -> str:
        latest_feedback = self._feedback(rounds[-1]) if rounds else "未能形成可靠审查记录。"
        return (
            "系统提示：针对该病例，AI 专家组未能形成足够安全的一致意见。"
            "本系统拒绝提供最终诊疗方案，请交由人类医生判断。\n\n"
            "分歧报告草稿：\n"
            f"{latest_feedback}"
        )

    @staticmethod
    def _with_disclaimer(text: str) -> str:
        return f"{DISCLAIMER}\n\n{text}"
