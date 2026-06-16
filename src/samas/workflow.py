from __future__ import annotations

from dataclasses import dataclass

from micro_mdt.agents import Agent
from medical_ttc.baselines import extract_mcq_letter
from medical_ttc.operators import EnsembleOperator, GenerateOperator, ReviewOperator, ReviseOperator
from medical_ttc.workflow import Workflow, WorkflowResult
from micro_mdt.models import PatientCase
from micro_mdt.providers import EmbeddingProvider, LLMProvider, MockEmbeddingProvider, MockProvider
from .controller import (
    OPERATOR_DESCRIPTIONS,
    SAMASOperatorController,
    SAMASTemplateController,
    OperatorPlanSelection,
    TemplateSelection,
)
from .profiler import Profile, PromptProfiler


MCQ_GENERALIST_PROMPT = """You are a careful medical exam solver in a research benchmark.
The input is a multiple-choice medical question. Choose exactly one option.

Rules:
1. Reason briefly from the clinical facts.
2. Do not provide real-world prescriptions beyond the answer choice context.
3. End with exactly: Final answer: <A/B/C/D>
"""


MCQ_SPECIALIST_PROMPT = """You are a medical specialist supporting a multiple-choice exam solver.
Give a concise specialist opinion and name the most likely option letter if possible.
End with: Final answer: <A/B/C/D>
"""


MCQ_MODERATOR_PROMPT = """You are a moderator for multiple medical expert opinions.
Synthesize the opinions and select exactly one best multiple-choice option.
End with exactly: Final answer: <A/B/C/D>
"""


TRIAGE_MCQ_PROMPT = """You are an emergency-aware medical exam solver.
This is a benchmark multiple-choice question. If the vignette describes an emergency,
choose the answer option that provides the safest urgent evaluation or treatment.
End with exactly: Final answer: <A/B/C/D>
"""


RX_MCQ_PROMPT = """You are a clinical pharmacology medical exam solver.
Focus on medication contraindications, interactions, pregnancy/child/renal warnings,
and the exact wording of the answer options.
End with exactly: Final answer: <A/B/C/D>
"""


REFUSAL_TEXT = (
    "This query appears unsafe for a medical QA system. I cannot provide operational "
    "harmful medical instructions. Final answer: ABSTAIN"
)


EDU_MCQ_PROMPT = """You are a patient-education oriented medical exam solver.
Explain briefly in plain language, but still choose exactly one benchmark option.
End with exactly: Final answer: <A/B/C/D>
"""


@dataclass(frozen=True)
class TemplateDecision:
    template_id: str
    reason: str


class RuleTemplateController:
    """Pilot controller: deterministic mapping from prompt profile to A1-A10.

    This is intentionally not the trained MaAS-style controller yet. It matches
    the current MedQA pilot scope: profiler-by-prompt first, one dataset first.
    """

    def select(self, profile: Profile) -> TemplateDecision:
        if profile.task_type == "ETHICS":
            return TemplateDecision("A10", "harmful/high-risk route")
        if profile.task_type == "TRIAGE":
            return TemplateDecision("A9", "emergency triage route")
        if profile.task_type == "RX":
            return TemplateDecision("A7", "medication route")
        if profile.complexity == "HIGH":
            return TemplateDecision("A6", "high complexity route")
        if profile.risk == "HIGH_RISK":
            return TemplateDecision("A3", "high-risk safety skeleton route")
        if profile.complexity == "MEDIUM" and profile.risk == "MODERATE":
            return TemplateDecision("A5", "medium risk route")
        if profile.task_type == "EDU":
            return TemplateDecision("A8", "education route")
        if profile.complexity == "MEDIUM":
            return TemplateDecision("A4", "medium safe route")
        return TemplateDecision("A1", "low safe route")


class SAMASPromptWorkflow(Workflow):
    """SAMAS-ZS pilot workflow for MedQA.

    The workflow implements:
        PromptProfiler(Q) -> rule template controller -> execute A-template

    Only the template subset needed to run MedQA is implemented in detail.
    A9/A10 exist so the profiler/controller semantics match the plan.
    """

    def __init__(self, provider: LLMProvider, *, enable_review: bool = False) -> None:
        super().__init__("SAMAS_ZS_MedQA_Pilot")
        self.provider = provider
        self.profiler = PromptProfiler(provider)
        self.controller = RuleTemplateController()
        self.enable_review = enable_review
        self.direct = GenerateOperator(provider, prompt=MCQ_GENERALIST_PROMPT)
        self.triage = GenerateOperator(provider, prompt=TRIAGE_MCQ_PROMPT)
        self.rx = GenerateOperator(provider, prompt=RX_MCQ_PROMPT)
        self.edu = GenerateOperator(provider, prompt=EDU_MCQ_PROMPT)
        self.ensemble = EnsembleOperator(provider, n_samples=3)
        self.review = ReviewOperator(provider)
        self.revise = ReviseOperator(provider)
        self.specialists = [
            Agent("CardiologySpecialist", MCQ_SPECIALIST_PROMPT, provider),
            Agent("PharmacySpecialist", RX_MCQ_PROMPT, provider),
            Agent("EmergencySpecialist", TRIAGE_MCQ_PROMPT, provider),
        ]
        self.moderator = Agent("SAMASModerator", MCQ_MODERATOR_PROMPT, provider)

    def __call__(self, case: PatientCase) -> WorkflowResult:
        profile = self.profiler(case.text)
        decision = self.controller.select(profile)
        return self._execute_decision(case, profile, decision)

    def _execute_decision(
        self,
        case: PatientCase,
        profile: Profile,
        decision: TemplateDecision | TemplateSelection,
        *,
        controller_metadata: dict | None = None,
    ) -> WorkflowResult:
        trace: list[dict] = [
            {
                "step": "profile",
                "complexity": profile.complexity,
                "risk": profile.risk,
                "task_type": profile.task_type,
                "rationale": profile.rationale,
            },
            {"step": "select_template", "template": decision.template_id, "reason": decision.reason},
        ]
        if isinstance(decision, TemplateSelection):
            trace[-1]["allowed_templates"] = list(decision.allowed_templates)
            trace[-1]["probs"] = decision.probs

        if decision.template_id == "A10":
            return WorkflowResult(
                case=case,
                final_answer=REFUSAL_TEXT,
                abstained=True,
                rounds=1,
                difficulty=profile.complexity,
                trace=trace,
                metadata={
                    "profile": profile.__dict__,
                    "template": decision.template_id,
                    **(controller_metadata or {}),
                },
            )

        if decision.template_id == "A8":
            answer = self.edu(case.text, temperature=0.0)
            trace.append({"step": "A8_explainer_answer", "answer": answer[:200]})
            return self._maybe_review(
                case,
                answer,
                profile,
                decision.template_id,
                trace,
                rounds=2,
                extra_metadata=controller_metadata,
            )

        if decision.template_id == "A9":
            answer = self.triage(case.text, temperature=0.0)
            trace.append({"step": "A9_triage_answer", "answer": answer[:200]})
            return self._maybe_review(
                case,
                answer,
                profile,
                decision.template_id,
                trace,
                rounds=2,
                extra_metadata=controller_metadata,
            )

        if decision.template_id == "A7":
            lookup = self._knowledge_lookup(case.text)
            answer = self.rx(f"{case.text}\n\nSafety lookup:\n{lookup}", temperature=0.0)
            trace.append({"step": "A7_lookup", "content": lookup})
            trace.append({"step": "A7_rx_answer", "answer": answer[:200]})
            return self._maybe_review(
                case,
                answer,
                profile,
                decision.template_id,
                trace,
                rounds=3,
                extra_metadata=controller_metadata,
            )

        if decision.template_id == "A4":
            answer, samples = self.ensemble(
                self.direct,
                case.text,
                extract_answer=extract_mcq_letter,
                temperature=0.5,
            )
            trace.append({"step": "A4_voter", "samples": len(samples), "answer": answer[:200]})
            return self._result(
                case,
                answer,
                profile,
                decision.template_id,
                trace,
                rounds=3,
                extra_metadata=controller_metadata,
            )

        if decision.template_id in {"A2", "A5", "A6"}:
            if decision.template_id == "A2":
                initial = self.direct(case.text, temperature=0.0)
                views = [initial]
                trace.append({"step": "A2_generalist", "answer": initial[:200]})
            else:
                n_specialists = 2 if decision.template_id == "A5" else 3
                views = [
                    agent.run(case.text, temperature=0.2).content
                    for agent in self.specialists[:n_specialists]
                ]
                trace.append({"step": f"{decision.template_id}_specialists", "n": len(views)})
            joined = "\n\n".join(f"Expert {i + 1}:\n{view}" for i, view in enumerate(views))
            answer = self.moderator.run(
                f"Question:\n{case.text}\n\nExpert opinions:\n{joined}",
                temperature=0.0,
            ).content
            trace.append({"step": f"{decision.template_id}_moderator", "answer": answer[:200]})
            if decision.template_id == "A2":
                return self._result(
                    case,
                    answer,
                    profile,
                    decision.template_id,
                    trace,
                    rounds=2,
                    extra_metadata=controller_metadata,
                )
            return self._maybe_review(
                case,
                answer,
                profile,
                decision.template_id,
                trace,
                rounds=4 if decision.template_id == "A5" else 5,
                extra_metadata=controller_metadata,
            )

        answer = self.direct(case.text, temperature=0.0)
        trace.append({"step": f"{decision.template_id}_direct", "answer": answer[:200]})
        if decision.template_id == "A3":
            return self._maybe_review(
                case,
                answer,
                profile,
                decision.template_id,
                trace,
                rounds=1,
                extra_metadata=controller_metadata,
            )
        return self._result(
            case,
            answer,
            profile,
            decision.template_id,
            trace,
            rounds=1,
            extra_metadata=controller_metadata,
        )

    def _maybe_review(
        self,
        case: PatientCase,
        answer: str,
        profile: Profile,
        template_id: str,
        trace: list[dict],
        *,
        rounds: int,
        extra_metadata: dict | None = None,
    ) -> WorkflowResult:
        if not self.enable_review:
            return self._result(
                case,
                answer,
                profile,
                template_id,
                trace,
                rounds=rounds,
                extra_metadata=extra_metadata,
            )
        verdicts = self.review(case.text, answer)
        trace.append(
            {
                "step": "review",
                "pharmacy": verdicts["pharmacy_verdict"].value,
                "safety": verdicts["safety_verdict"].value,
            }
        )
        if verdicts["abstain"]:
            return self._result(
                case,
                (
                    "Safety review requested human takeover for this item. "
                    "Final answer: ABSTAIN"
                ),
                profile,
                template_id,
                trace,
                rounds=rounds + 1,
                abstained=True,
                extra_metadata=extra_metadata,
            )
        if verdicts["pass"]:
            return self._result(
                case,
                answer,
                profile,
                template_id,
                trace,
                rounds=rounds + 1,
                extra_metadata=extra_metadata,
            )
        feedback = f"Pharmacy: {verdicts['pharmacy_content']}\nSafety: {verdicts['safety_content']}"
        revised = self.revise(case.text, answer, feedback)
        trace.append({"step": "revise", "answer": revised[:200]})
        return self._result(
            case,
            revised,
            profile,
            template_id,
            trace,
            rounds=rounds + 2,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def _result(
        case: PatientCase,
        answer: str,
        profile: Profile,
        template_id: str,
        trace: list[dict],
        *,
        rounds: int,
        abstained: bool = False,
        extra_metadata: dict | None = None,
    ) -> WorkflowResult:
        metadata = {
            "profile": {
                "complexity": profile.complexity,
                "risk": profile.risk,
                "task_type": profile.task_type,
                "rationale": profile.rationale,
                "raw": profile.raw[:500],
            },
            "template": template_id,
        }
        metadata.update(extra_metadata or {})
        return WorkflowResult(
            case=case,
            final_answer=answer,
            abstained=abstained,
            rounds=rounds,
            difficulty=profile.complexity,
            trace=trace,
            metadata=metadata,
        )

    @staticmethod
    def _knowledge_lookup(query: str) -> str:
        lower = query.lower()
        findings: list[str] = []
        if "warfarin" in lower and ("ibuprofen" in lower or "aspirin" in lower):
            findings.append("warfarin plus NSAID/aspirin can increase bleeding risk")
        if "metformin" in lower and ("egfr 25" in lower or "renal" in lower or "kidney" in lower):
            findings.append("metformin is commonly avoided or contraindicated in severe renal impairment")
        if "lisinopril" in lower and ("pregnant" in lower or "pregnancy" in lower or "gestation" in lower):
            findings.append("ACE inhibitors are contraindicated in pregnancy")
        if "chest pain" in lower or "st elevation" in lower or "呼吸困难" in lower or "胸痛" in lower:
            findings.append("red flag: possible emergency, prioritize urgent evaluation/treatment options")
        if not findings:
            findings.append("no table-based drug or red-flag warning matched")
        return "; ".join(findings)


class SAMASControllerWorkflow(SAMASPromptWorkflow):
    """SAMAS-ZS with a learned MaAS-style template controller.

    The profiler remains prompt-based; the architecture selector is a trainable
    policy over complete templates A1-A10.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        controller: SAMASTemplateController | None = None,
        sample: bool = False,
        enable_review: bool = True,
    ) -> None:
        super().__init__(provider, enable_review=enable_review)
        self.name = "SAMAS_ZS_Controller"
        self.controller = controller or SAMASTemplateController()
        self.sample = sample

    def __call__(self, case: PatientCase) -> WorkflowResult:
        profile = self.profiler(case.text)
        decision = self.controller.select_template(case.text, profile, sample=self.sample)
        metadata = {
            "controller": {
                "type": "SAMASTemplateController",
                "sample": self.sample,
                "allowed_templates": list(decision.allowed_templates),
                "probs": decision.probs,
            }
        }
        if decision.log_prob is not None:
            metadata["controller"]["log_prob"] = float(decision.log_prob.detach().cpu().item())
        return self._execute_decision(case, profile, decision, controller_metadata=metadata)


class SAMASOperatorWorkflow(SAMASPromptWorkflow):
    """SAMAS with a step-wise MaAS controller over operators.

    The prompt profiler is still zero-shot, but the controller no longer
    classifies each query into one full template. It builds a workflow by
    selecting one operator at a time, executing the operator, and feeding the
    resulting state into the next action decision.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        controller: SAMASOperatorController | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        sample: bool = False,
    ) -> None:
        super().__init__(provider, enable_review=False)
        self.name = "SAMAS_ZS_OperatorController"
        self.controller = controller or SAMASOperatorController()
        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else (MockEmbeddingProvider(dim=self.controller.semantic_dim) if isinstance(provider, MockProvider) else None)
        )
        self._sync_operator_embeddings()
        self.sample = sample

    def __call__(self, case: PatientCase) -> WorkflowResult:
        profile = self.profiler(case.text)
        result, _ = self._execute_interleaved(case, profile, sample=self.sample)
        return result

    def _execute_interleaved(
        self,
        case: PatientCase,
        profile: Profile,
        *,
        sample: bool,
    ) -> tuple[WorkflowResult, OperatorPlanSelection]:
        runtime = {
            "lookup_text": "",
            "views": [],
            "draft": "",
            "review_cache": None,
            "feedback": [],
        }
        controller_state = self.controller.initial_state()
        steps = []
        operators: list[str] = []
        log_probs = []
        rounds = 0
        trace: list[dict] = [
            {
                "step": "profile",
                "complexity": profile.complexity,
                "risk": profile.risk,
                "task_type": profile.task_type,
                "rationale": profile.rationale,
            },
            {"step": "operator_controller", "mode": "interleaved_select_execute"},
        ]

        def current_plan() -> OperatorPlanSelection:
            log_prob = None
            if log_probs:
                import torch

                log_prob = torch.stack(log_probs).sum()
            layers: list[list[str]] = []
            for step in steps:
                while len(layers) <= step.layer:
                    layers.append([])
                layers[step.layer].append(step.operator_id)
            return OperatorPlanSelection(
                operators=list(operators),
                steps=list(steps),
                log_prob=log_prob,
                layers=layers,
            )

        for step_idx in range(self.controller.max_steps):
            self._sync_controller_semantic_state(case.text, runtime, controller_state)
            selected = self.controller.select_next(
                case.text,
                profile,
                controller_state,
                step=step_idx,
                sample=sample,
            )
            op = selected.operator_id
            steps.append(selected)
            operators.append(op)
            if selected.log_prob is not None:
                log_probs.append(selected.log_prob)
            trace.append(
                {
                    "step": "controller_step",
                    "index": step_idx,
                    "layer": selected.layer,
                    "operator": op,
                    "state": {
                        "has_lookup": bool(controller_state.get("has_lookup")),
                        "has_draft": bool(controller_state.get("has_draft")),
                        "n_views": int(controller_state.get("n_views", 0)),
                        "has_pharmacy": bool(controller_state.get("has_pharmacy")),
                        "has_safety": bool(controller_state.get("has_safety")),
                        "has_revised": bool(controller_state.get("has_revised")),
                        "semantic_nonzero": any(abs(float(v)) > 1e-9 for v in controller_state.get("semantic", [])),
                    },
                    "allowed_operators": list(selected.allowed_operators),
                    "probs": selected.probs,
                }
            )

            if op == "ABSTAIN":
                self.controller.update_planning_state(controller_state, op)
                trace.append({"step": "operator", "operator": op})
                plan = current_plan()
                return (
                    self._operator_result(
                        case,
                        REFUSAL_TEXT,
                        profile,
                        plan,
                        trace,
                        rounds=max(rounds, 1),
                        abstained=True,
                    ),
                    plan,
                )

            if op == "STOP":
                self.controller.update_planning_state(controller_state, op)
                trace.append({"step": "operator", "operator": op})
                final = runtime["draft"] or self._safe_fallback_answer(case, profile, runtime, trace)
                if not runtime["draft"]:
                    rounds += 1
                plan = current_plan()
                return (
                    self._operator_result(case, final, profile, plan, trace, rounds=max(rounds, 1)),
                    plan,
                )

            if op == "LOOKUP":
                runtime["lookup_text"] = self._knowledge_lookup(case.text)
                trace.append({"step": "operator", "operator": op, "content": runtime["lookup_text"]})
                self.controller.update_planning_state(controller_state, op)
                continue

            if op == "DIRECT":
                runtime["draft"] = self.direct(self._with_context(case.text, runtime), temperature=0.0)
                rounds += 1
                trace.append({"step": "operator", "operator": op, "answer": runtime["draft"][:200]})
                self.controller.update_planning_state(controller_state, op)
                continue

            if op == "GENERALIST":
                view = self.direct(self._with_context(case.text, runtime), temperature=0.2)
                runtime["views"].append(view)
                rounds += 1
                trace.append({"step": "operator", "operator": op, "view": view[:200]})
                self.controller.update_planning_state(controller_state, op)
                continue

            if op == "SPECIALIST":
                specialist = self._select_specialist(profile, len(runtime["views"]))
                view = specialist.run(self._with_context(case.text, runtime), temperature=0.2).content
                runtime["views"].append(view)
                rounds += 1
                trace.append(
                    {
                        "step": "operator",
                        "operator": op,
                        "agent": specialist.name,
                        "view": view[:200],
                    }
                )
                self.controller.update_planning_state(controller_state, op)
                continue

            if op == "VOTER":
                answer, samples = self.ensemble(
                    self.direct,
                    self._with_context(case.text, runtime),
                    extract_answer=extract_mcq_letter,
                    temperature=0.5,
                )
                runtime["draft"] = answer
                rounds += len(samples)
                trace.append(
                    {
                        "step": "operator",
                        "operator": op,
                        "samples": len(samples),
                        "answer": answer[:200],
                    }
                )
                self.controller.update_planning_state(controller_state, op)
                continue

            if op == "MODERATOR":
                views = list(runtime["views"])
                if runtime["draft"]:
                    views.append(runtime["draft"])
                if not views:
                    runtime["draft"] = self.direct(self._with_context(case.text, runtime), temperature=0.0)
                    rounds += 1
                    trace.append(
                        {
                            "step": "operator",
                            "operator": op,
                            "fallback": "direct_before_moderator",
                            "answer": runtime["draft"][:200],
                        }
                    )
                    self.controller.update_planning_state(controller_state, op)
                    continue
                joined = "\n\n".join(f"Expert {i + 1}:\n{view}" for i, view in enumerate(views))
                runtime["draft"] = self.moderator.run(
                    f"Question:\n{case.text}\n\nExpert opinions:\n{joined}",
                    temperature=0.0,
                ).content
                rounds += 1
                trace.append(
                    {
                        "step": "operator",
                        "operator": op,
                        "n_views": len(views),
                        "answer": runtime["draft"][:200],
                    }
                )
                self.controller.update_planning_state(controller_state, op)
                continue

            if op in {"PHARMACY", "SAFETY"}:
                review_rounds = self._ensure_review(case, runtime)
                rounds += review_rounds
                verdicts = runtime["review_cache"]
                if op == "PHARMACY":
                    runtime["feedback"].append(f"Pharmacy: {verdicts['pharmacy_content']}")
                    trace.append(
                        {
                            "step": "operator",
                            "operator": op,
                            "verdict": verdicts["pharmacy_verdict"].value,
                        }
                    )
                else:
                    runtime["feedback"].append(f"Safety: {verdicts['safety_content']}")
                    trace.append(
                        {
                            "step": "operator",
                            "operator": op,
                            "verdict": verdicts["safety_verdict"].value,
                        }
                    )
                self.controller.update_planning_state(controller_state, op)
                if verdicts["abstain"]:
                    plan = current_plan()
                    return (
                        self._operator_result(
                            case,
                            "Safety review requested human takeover for this item. Final answer: ABSTAIN",
                            profile,
                            plan,
                            trace,
                            rounds=max(rounds, 1),
                            abstained=True,
                        ),
                        plan,
                    )
                continue

            if op == "REVISER":
                feedback = "\n".join(runtime["feedback"]) or "Improve medical safety and answer format."
                runtime["draft"] = self.revise(case.text, runtime["draft"], feedback)
                rounds += 1
                trace.append({"step": "operator", "operator": op, "answer": runtime["draft"][:200]})
                self.controller.update_planning_state(controller_state, op)

        plan = current_plan()
        if runtime["draft"]:
            return (
                self._operator_result(case, runtime["draft"], profile, plan, trace, rounds=max(rounds, 1)),
                plan,
            )
        abstain = profile.risk == "HIGH_RISK" or profile.task_type == "ETHICS"
        final = REFUSAL_TEXT if abstain else self._safe_fallback_answer(case, profile, runtime, trace)
        if not abstain:
            rounds += 1
        return (
            self._operator_result(
                case,
                final,
                profile,
                plan,
                trace,
                rounds=max(rounds, 1),
                abstained=abstain,
            ),
            plan,
        )

    def _execute_operator_plan(
        self,
        case: PatientCase,
        profile: Profile,
        plan: OperatorPlanSelection,
    ) -> WorkflowResult:
        runtime = {
            "lookup_text": "",
            "views": [],
            "draft": "",
            "review_cache": None,
            "feedback": [],
        }
        rounds = 0
        trace: list[dict] = [
            {
                "step": "profile",
                "complexity": profile.complexity,
                "risk": profile.risk,
                "task_type": profile.task_type,
                "rationale": profile.rationale,
            },
            {
                "step": "select_operator_plan",
                "operators": plan.operators,
                "steps": [
                    {
                        "step": step.step,
                        "operator": step.operator_id,
                        "allowed_operators": list(step.allowed_operators),
                        "probs": step.probs,
                    }
                    for step in plan.steps
                ],
            },
        ]

        for op in plan.operators:
            if op == "ABSTAIN":
                trace.append({"step": "operator", "operator": op})
                return self._operator_result(
                    case,
                    REFUSAL_TEXT,
                    profile,
                    plan,
                    trace,
                    rounds=max(rounds, 1),
                    abstained=True,
                )

            if op == "STOP":
                trace.append({"step": "operator", "operator": op})
                final = runtime["draft"] or self._safe_fallback_answer(case, profile, runtime, trace)
                if not runtime["draft"]:
                    rounds += 1
                return self._operator_result(case, final, profile, plan, trace, rounds=max(rounds, 1))

            if op == "LOOKUP":
                runtime["lookup_text"] = self._knowledge_lookup(case.text)
                trace.append({"step": "operator", "operator": op, "content": runtime["lookup_text"]})
                continue

            if op == "DIRECT":
                runtime["draft"] = self.direct(self._with_context(case.text, runtime), temperature=0.0)
                rounds += 1
                trace.append({"step": "operator", "operator": op, "answer": runtime["draft"][:200]})
                continue

            if op == "GENERALIST":
                view = self.direct(self._with_context(case.text, runtime), temperature=0.2)
                runtime["views"].append(view)
                rounds += 1
                trace.append({"step": "operator", "operator": op, "view": view[:200]})
                continue

            if op == "SPECIALIST":
                specialist = self._select_specialist(profile, len(runtime["views"]))
                view = specialist.run(self._with_context(case.text, runtime), temperature=0.2).content
                runtime["views"].append(view)
                rounds += 1
                trace.append(
                    {
                        "step": "operator",
                        "operator": op,
                        "agent": specialist.name,
                        "view": view[:200],
                    }
                )
                continue

            if op == "VOTER":
                answer, samples = self.ensemble(
                    self.direct,
                    self._with_context(case.text, runtime),
                    extract_answer=extract_mcq_letter,
                    temperature=0.5,
                )
                runtime["draft"] = answer
                rounds += len(samples)
                trace.append(
                    {
                        "step": "operator",
                        "operator": op,
                        "samples": len(samples),
                        "answer": answer[:200],
                    }
                )
                continue

            if op == "MODERATOR":
                views = list(runtime["views"])
                if runtime["draft"]:
                    views.append(runtime["draft"])
                if not views:
                    runtime["draft"] = self.direct(self._with_context(case.text, runtime), temperature=0.0)
                    rounds += 1
                    trace.append(
                        {
                            "step": "operator",
                            "operator": op,
                            "fallback": "direct_before_moderator",
                            "answer": runtime["draft"][:200],
                        }
                    )
                    continue
                joined = "\n\n".join(f"Expert {i + 1}:\n{view}" for i, view in enumerate(views))
                runtime["draft"] = self.moderator.run(
                    f"Question:\n{case.text}\n\nExpert opinions:\n{joined}",
                    temperature=0.0,
                ).content
                rounds += 1
                trace.append(
                    {
                        "step": "operator",
                        "operator": op,
                        "n_views": len(views),
                        "answer": runtime["draft"][:200],
                    }
                )
                continue

            if op in {"PHARMACY", "SAFETY"}:
                if not runtime["draft"]:
                    runtime["draft"] = self.direct(self._with_context(case.text, runtime), temperature=0.0)
                    rounds += 1
                    trace.append(
                        {
                            "step": "operator",
                            "operator": op,
                            "fallback": "direct_before_review",
                            "answer": runtime["draft"][:200],
                        }
                    )
                review_rounds = self._ensure_review(case, runtime)
                rounds += review_rounds
                verdicts = runtime["review_cache"]
                if op == "PHARMACY":
                    runtime["feedback"].append(f"Pharmacy: {verdicts['pharmacy_content']}")
                    trace.append(
                        {
                            "step": "operator",
                            "operator": op,
                            "verdict": verdicts["pharmacy_verdict"].value,
                        }
                    )
                else:
                    runtime["feedback"].append(f"Safety: {verdicts['safety_content']}")
                    trace.append(
                        {
                            "step": "operator",
                            "operator": op,
                            "verdict": verdicts["safety_verdict"].value,
                        }
                    )
                if verdicts["abstain"]:
                    return self._operator_result(
                        case,
                        "Safety review requested human takeover for this item. Final answer: ABSTAIN",
                        profile,
                        plan,
                        trace,
                        rounds=max(rounds, 1),
                        abstained=True,
                    )
                continue

            if op == "REVISER":
                if not runtime["draft"]:
                    runtime["draft"] = self.direct(self._with_context(case.text, runtime), temperature=0.0)
                    rounds += 1
                feedback = "\n".join(runtime["feedback"]) or "Improve medical safety and answer format."
                runtime["draft"] = self.revise(case.text, runtime["draft"], feedback)
                rounds += 1
                trace.append({"step": "operator", "operator": op, "answer": runtime["draft"][:200]})

        if runtime["draft"]:
            return self._operator_result(case, runtime["draft"], profile, plan, trace, rounds=max(rounds, 1))
        abstain = profile.risk == "HIGH_RISK" or profile.task_type == "ETHICS"
        final = REFUSAL_TEXT if abstain else self._safe_fallback_answer(case, profile, runtime, trace)
        if not abstain:
            rounds += 1
        return self._operator_result(
            case,
            final,
            profile,
            plan,
            trace,
            rounds=max(rounds, 1),
            abstained=abstain,
        )

    def _with_context(self, question: str, runtime: dict) -> str:
        parts = [question]
        if runtime.get("lookup_text"):
            parts.append(f"Safety lookup:\n{runtime['lookup_text']}")
        if runtime.get("draft"):
            parts.append(f"Current draft:\n{runtime['draft']}")
        return "\n\n".join(parts)

    def _sync_controller_semantic_state(self, question: str, runtime: dict, controller_state: dict) -> None:
        if self.embedding_provider is None:
            controller_state["semantic"] = [0.0] * self.controller.semantic_dim
            return
        text = self._semantic_state_text(question, runtime)
        embedding = self.embedding_provider.embed(text)
        controller_state["semantic"] = self._compress_embedding(embedding, self.controller.semantic_dim)

    def _sync_operator_embeddings(self) -> None:
        if self.embedding_provider is None:
            return
        embeddings = {
            op: self._compress_embedding(
                self.embedding_provider.embed(description),
                self.controller.operator_embedding_dim,
            )
            for op, description in OPERATOR_DESCRIPTIONS.items()
        }
        self.controller.set_operator_embeddings(embeddings)

    @staticmethod
    def _semantic_state_text(question: str, runtime: dict) -> str:
        parts = [f"Question:\n{question}"]
        if runtime.get("lookup_text"):
            parts.append(f"Lookup:\n{runtime['lookup_text']}")
        views = runtime.get("views") or []
        for i, view in enumerate(views[-3:]):
            parts.append(f"View {i + 1}:\n{view}")
        if runtime.get("draft"):
            parts.append(f"Draft:\n{runtime['draft']}")
        feedback = runtime.get("feedback") or []
        if feedback:
            parts.append("Review feedback:\n" + "\n".join(feedback[-4:]))
        return "\n\n".join(parts)[-8192:]

    @staticmethod
    def _compress_embedding(values: list[float], target_dim: int) -> list[float]:
        if target_dim <= 0:
            return []
        if not values:
            return [0.0] * target_dim
        buckets = [0.0] * target_dim
        for i, value in enumerate(values):
            buckets[i % target_dim] += float(value)
        scale = max(1, len(values) // target_dim)
        buckets = [v / scale for v in buckets]
        norm = sum(v * v for v in buckets) ** 0.5
        if norm > 0:
            buckets = [v / norm for v in buckets]
        return buckets

    def _select_specialist(self, profile: Profile, index: int) -> Agent:
        if profile.task_type == "RX":
            return self.specialists[1]
        if profile.task_type == "TRIAGE":
            return self.specialists[2]
        return self.specialists[index % len(self.specialists)]

    def _ensure_review(self, case: PatientCase, runtime: dict) -> int:
        if runtime.get("review_cache") is not None:
            return 0
        runtime["review_cache"] = self.review(case.text, runtime["draft"])
        return 2

    def _safe_fallback_answer(
        self,
        case: PatientCase,
        profile: Profile,
        runtime: dict,
        trace: list[dict],
    ) -> str:
        if profile.task_type == "RX":
            answer = self.rx(self._with_context(case.text, runtime), temperature=0.0)
        elif profile.task_type == "TRIAGE":
            answer = self.triage(self._with_context(case.text, runtime), temperature=0.0)
        elif profile.task_type == "EDU":
            answer = self.edu(self._with_context(case.text, runtime), temperature=0.0)
        else:
            answer = self.direct(self._with_context(case.text, runtime), temperature=0.0)
        trace.append({"step": "fallback_answer", "answer": answer[:200]})
        return answer

    @staticmethod
    def _operator_result(
        case: PatientCase,
        answer: str,
        profile: Profile,
        plan: OperatorPlanSelection,
        trace: list[dict],
        *,
        rounds: int,
        abstained: bool = False,
    ) -> WorkflowResult:
        metadata = {
            "profile": {
                "complexity": profile.complexity,
                "risk": profile.risk,
                "task_type": profile.task_type,
                "rationale": profile.rationale,
                "raw": profile.raw[:500],
            },
            "controller": {
                "type": "SAMASOperatorController",
                "selection": "MaAS-style multilayer operator policy",
                "operators": plan.operators,
                "layers": plan.layers or [],
            },
        }
        if plan.log_prob is not None:
            metadata["controller"]["log_prob"] = float(plan.log_prob.detach().cpu().item())
        return WorkflowResult(
            case=case,
            final_answer=answer,
            abstained=abstained,
            rounds=rounds,
            difficulty=profile.complexity,
            trace=trace,
            metadata=metadata,
        )
