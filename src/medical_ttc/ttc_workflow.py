"""W4 — Medical TTC main workflow.

Combines:
    - Router (MaAS-inspired): decides LOW/MED/HIGH compute tier per case
    - Per-tier workflow body:
          LOW  → Direct generate
          MED  → Propose → Review → (optionally Revise once)
          HIGH → MDT-style debate (≤3 rounds: Propose → Review → Revise)
    - Hard safety constraint (SafetyEthics + Pharmacist verdicts can ABSTAIN)
    - Compatible with the AFLOW MCTS optimizer — the optimizer can mutate
      the prompt templates used inside each tier without violating the
      safety骨架.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from micro_mdt.models import Difficulty, PatientCase
from micro_mdt.providers import LLMProvider
from .operators import (
    EnsembleOperator,
    GenerateOperator,
    ModeratorOperator,
    ReviewOperator,
    ReviseOperator,
    TriageOperator,
)
from .router import ZeroShotRouter
from .workflow import Workflow, WorkflowResult


@dataclass
class TierPrompts:
    """Per-tier prompt overrides. Optimizer can mutate these."""

    low_generate: str | None = None
    med_generate: str | None = None
    med_revise: str | None = None
    high_generate: str | None = None
    high_revise: str | None = None
    high_moderator: str | None = None
    ensemble_n: int = 3  # number of high-tier ensemble samples


class MedicalTTCWorkflow(Workflow):
    """The W4 main workflow."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        router=None,
        max_high_rounds: int = 3,
        enable_safety: bool = True,
        prompts: TierPrompts | None = None,
        name: str = "W4_MedicalTTC",
    ) -> None:
        super().__init__(name)
        self.provider = provider
        self.router = router or ZeroShotRouter(provider)
        self.max_high_rounds = max_high_rounds
        self.enable_safety = enable_safety
        p = prompts or TierPrompts()
        self.prompts = p
        # Build operators
        self.gen_low = GenerateOperator(provider, prompt=p.low_generate)
        self.gen_med = GenerateOperator(provider, prompt=p.med_generate)
        self.gen_high = GenerateOperator(provider, prompt=p.high_generate)
        self.review = ReviewOperator(provider)
        self.revise_med = ReviseOperator(provider, prompt=p.med_revise)
        self.revise_high = ReviseOperator(provider, prompt=p.high_revise)
        self.moderator = ModeratorOperator(provider, prompt=p.high_moderator)
        self.ensemble = EnsembleOperator(provider, n_samples=p.ensemble_n)

    # ----------------------------- entry -----------------------------

    def __call__(self, case: PatientCase) -> WorkflowResult:
        difficulty = self.router.route(case.text)
        trace: list[dict] = [{"step": "route", "difficulty": difficulty.value}]
        if difficulty is Difficulty.LOW:
            return self._tier_low(case, trace)
        if difficulty is Difficulty.MEDIUM:
            return self._tier_med(case, trace)
        return self._tier_high(case, trace)

    # ----------------------------- tiers -----------------------------

    def _tier_low(self, case: PatientCase, trace: list[dict]) -> WorkflowResult:
        answer = self.gen_low(case.text)
        trace.append({"step": "low_generate", "out": answer[:160]})
        # No safety audit on LOW (these are deemed routine queries)
        return WorkflowResult(
            case=case,
            final_answer=answer,
            rounds=1,
            difficulty=Difficulty.LOW.value,
            trace=trace,
        )

    def _tier_med(self, case: PatientCase, trace: list[dict]) -> WorkflowResult:
        proposal = self.gen_med(case.text)
        trace.append({"step": "med_propose", "out": proposal[:160]})
        if self.enable_safety:
            verdicts = self.review(case.text, proposal)
            trace.append(
                {
                    "step": "med_review",
                    "pharm": verdicts["pharmacy_verdict"].value,
                    "safety": verdicts["safety_verdict"].value,
                }
            )
            if verdicts["abstain"]:
                return WorkflowResult(
                    case=case,
                    final_answer=self._abstain_msg(verdicts),
                    abstained=True,
                    rounds=2,
                    difficulty=Difficulty.MEDIUM.value,
                    trace=trace,
                )
            if not verdicts["pass"]:
                feedback = self._format_feedback(verdicts)
                proposal = self.revise_med(case.text, proposal, feedback)
                trace.append({"step": "med_revise", "out": proposal[:160]})
        return WorkflowResult(
            case=case,
            final_answer=proposal,
            rounds=2 if self.enable_safety else 1,
            difficulty=Difficulty.MEDIUM.value,
            trace=trace,
        )

    def _tier_high(self, case: PatientCase, trace: list[dict]) -> WorkflowResult:
        # 1. Ensemble propose (multiple perspectives via temperature variance)
        winner, samples = self.ensemble(self.gen_high, case.text)
        trace.append(
            {"step": "high_ensemble", "n_samples": len(samples), "winner": winner[:160]}
        )
        # 2. Moderator synthesizes ensemble views
        synthesized = self.moderator(case.text, samples)
        trace.append({"step": "high_moderator", "out": synthesized[:160]})
        # 3. Up to N rounds of safety review + revise
        proposal = synthesized
        round_idx = 0
        if self.enable_safety:
            for round_idx in range(1, self.max_high_rounds + 1):
                verdicts = self.review(case.text, proposal)
                trace.append(
                    {
                        "step": f"high_review_r{round_idx}",
                        "pharm": verdicts["pharmacy_verdict"].value,
                        "safety": verdicts["safety_verdict"].value,
                    }
                )
                if verdicts["pass"]:
                    break
                if verdicts["abstain"]:
                    return WorkflowResult(
                        case=case,
                        final_answer=self._abstain_msg(verdicts),
                        abstained=True,
                        rounds=2 + round_idx,
                        difficulty=Difficulty.HIGH.value,
                        trace=trace,
                    )
                feedback = self._format_feedback(verdicts)
                proposal = self.revise_high(case.text, proposal, feedback)
                trace.append({"step": f"high_revise_r{round_idx}", "out": proposal[:160]})
            else:
                # exhausted rounds without PASS — abstain (Snell-style verifier rejection)
                return WorkflowResult(
                    case=case,
                    final_answer=(
                        "⚠️ 多轮审查未达成共识，触发拒答。请由人类医生接管。"
                    ),
                    abstained=True,
                    rounds=2 + self.max_high_rounds,
                    difficulty=Difficulty.HIGH.value,
                    trace=trace,
                )
        return WorkflowResult(
            case=case,
            final_answer=proposal,
            rounds=2 + round_idx if self.enable_safety else 2,
            difficulty=Difficulty.HIGH.value,
            trace=trace,
        )

    # --------------------------- helpers ----------------------------

    @staticmethod
    def _format_feedback(verdicts: dict) -> str:
        return (
            f"药师审查 ({verdicts['pharmacy_verdict'].value}): "
            f"{verdicts['pharmacy_content']}\n"
            f"安全审查 ({verdicts['safety_verdict'].value}): "
            f"{verdicts['safety_content']}"
        )

    @staticmethod
    def _abstain_msg(verdicts: dict) -> str:
        return (
            "⚠️ 安全审查触发拒答：\n"
            f"  药师: {verdicts['pharmacy_content'][:160]}\n"
            f"  安全: {verdicts['safety_content'][:160]}\n\n"
            "请由人类医生接管。"
        )
