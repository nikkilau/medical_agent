from __future__ import annotations

from ..models import AgentResponse, Difficulty, MDTResult, PatientCase
from ..profiler import HeuristicMedicalProfiler, MedicalProfile
from ..prompts import DISCLAIMER
from ..providers import LLMProvider
from ..router import DifficultySafetyRouter, TTCRoute
from .accuracy import AccuracyPath
from .dynamic_mdt import DynamicMDTPath
from .safety_review import SafetyReviewPath


class SafetyAwareMicroMDT:
    """Difficulty × safety TTC orchestrator for the first runnable prototype."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        low_safety_rounds: int = 1,
        high_safety_rounds: int = 3,
    ) -> None:
        self.provider = provider
        self.profiler = HeuristicMedicalProfiler()
        self.router = DifficultySafetyRouter()
        self.accuracy = AccuracyPath(provider)
        self.dynamic_mdt = DynamicMDTPath(provider)
        self.low_safety_rounds = low_safety_rounds
        self.high_safety_rounds = high_safety_rounds

    def run_case(self, patient_case: PatientCase, *, human_decision_callback=None) -> MDTResult:
        profile = self.profiler(patient_case.text)
        route = self.router.route(profile)
        trace = [
            AgentResponse(
                "MedicalProfiler",
                f"difficulty={profile.difficulty}, safety_risk={profile.safety_risk}; {profile.rationale}",
            ),
            AgentResponse("DifficultySafetyRouter", f"{route.name}: {route.reason}"),
        ]

        if route.name == "ACCURACY":
            answer, path_trace, rounds = self.accuracy.run(patient_case)
            trace.extend(path_trace)
            return self._result(patient_case, profile, route, "ttc_accuracy_completed", answer, trace, rounds)

        if route.name == "DYNAMIC_MDT":
            answer, path_trace, rounds = self.dynamic_mdt.run(patient_case)
            trace.extend(path_trace)
            return self._result(patient_case, profile, route, "ttc_dynamic_mdt_completed", answer, trace, rounds)

        max_rounds = self.high_safety_rounds if route.name == "SAFETY_REVIEW_HIGH" else self.low_safety_rounds
        safety_path = SafetyReviewPath(self.provider, max_rounds=max_rounds)
        answer, debate_rounds, path_trace, abstained, rounds = safety_path.run(patient_case)
        trace.extend(path_trace)
        status = "ttc_safety_abstained" if abstained else "ttc_safety_reviewed"
        return self._result(
            patient_case,
            profile,
            route,
            status,
            answer,
            trace,
            rounds,
            debate_rounds=debate_rounds,
            human_required=False,
        )

    def _result(
        self,
        case: PatientCase,
        profile: MedicalProfile,
        route: TTCRoute,
        status: str,
        answer: str,
        trace: list[AgentResponse],
        rounds: int,
        *,
        debate_rounds=None,
        human_required: bool = False,
    ) -> MDTResult:
        difficulty = Difficulty.HIGH if profile.difficulty == "COMPLEX" else Difficulty.LOW
        return MDTResult(
            case=case,
            difficulty=difficulty,
            status=status,
            final_answer=f"{DISCLAIMER}\n\n{answer}",
            rounds=debate_rounds or [],
            human_required=human_required,
            trace=trace,
            metadata={
                "profile": {
                    "difficulty": profile.difficulty,
                    "safety_risk": profile.safety_risk,
                    "rationale": profile.rationale,
                },
                "route": {"name": route.name, "reason": route.reason},
                "ttc_rounds": rounds,
            },
        )
