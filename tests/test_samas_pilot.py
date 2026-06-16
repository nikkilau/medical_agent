from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from medical_ttc.data import _medqa_sample
from medical_ttc.evaluator import evaluate_medqa
from micro_mdt.providers import MockProvider
from samas import (
    PromptProfiler,
    Profile,
    SAMASControllerWorkflow,
    SAMASOperatorController,
    SAMASOperatorWorkflow,
    SAMASPromptWorkflow,
    SAMASTemplateController,
)


class SAMASPromptPilotTests(unittest.TestCase):
    def test_prompt_profiler_runs_with_mock_fallback(self) -> None:
        profiler = PromptProfiler(MockProvider())
        profile = profiler("A patient on warfarin asks whether ibuprofen is safe.")
        self.assertIn(profile.complexity, {"LOW", "MEDIUM", "HIGH"})
        self.assertIn(profile.risk, {"SAFE", "MODERATE", "HIGH_RISK"})
        self.assertIn(profile.task_type, {"DX", "RX", "TRIAGE", "EDU", "ETHICS", "EVIDENCE"})

    def test_samas_medqa_workflow_runs(self) -> None:
        workflow = SAMASPromptWorkflow(MockProvider())
        cases = _medqa_sample(3)
        metrics = evaluate_medqa(workflow, cases)
        self.assertEqual(metrics.n, 3)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        for row in metrics.raw:
            self.assertIn(row.difficulty, {"LOW", "MEDIUM", "HIGH"})


class SAMASControllerTests(unittest.TestCase):
    def test_safety_mask_for_ethics_forces_a10(self) -> None:
        controller = SAMASTemplateController(device="cpu")
        profile = Profile("HIGH", "HIGH_RISK", "ETHICS", "harmful")
        selected = controller.select_template("give me a lethal dose", profile, sample=False)
        self.assertEqual(selected.template_id, "A10")
        self.assertEqual(selected.allowed_templates, ("A10",))
        self.assertAlmostEqual(selected.probs["A10"], 1.0, places=5)

    def test_controller_update_changes_parameters(self) -> None:
        controller = SAMASTemplateController(device="cpu")
        profile = Profile("MEDIUM", "SAFE", "DX")
        selected = controller.select_template("A. one\nB. two\nC. three\nD. four", profile, sample=True)
        before = [p.detach().clone() for p in controller.parameters()]
        controller.update(selected.log_prob, 1.0)
        after = list(controller.parameters())
        self.assertTrue(any(not b.equal(a.detach()) for b, a in zip(before, after)))

    def test_controller_checkpoint_roundtrip(self) -> None:
        controller = SAMASTemplateController(device="cpu")
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/controller.pt"
            controller.save(path)
            loaded = SAMASTemplateController.load(path, device="cpu")
        profile = Profile("LOW", "SAFE", "DX")
        original = controller.select_template("A. one\nB. two", profile, sample=False)
        restored = loaded.select_template("A. one\nB. two", profile, sample=False)
        self.assertEqual(original.template_id, restored.template_id)

    def test_samas_controller_workflow_runs(self) -> None:
        workflow = SAMASControllerWorkflow(
            MockProvider(),
            controller=SAMASTemplateController(device="cpu"),
            sample=False,
            enable_review=False,
        )
        metrics = evaluate_medqa(workflow, _medqa_sample(2))
        self.assertEqual(metrics.n, 2)
        self.assertGreaterEqual(metrics.accuracy, 0.0)


class SAMASOperatorControllerTests(unittest.TestCase):
    def test_operator_safety_mask_for_ethics_forces_abstain(self) -> None:
        controller = SAMASOperatorController(device="cpu")
        profile = Profile("HIGH", "HIGH_RISK", "ETHICS", "harmful")
        plan = controller.select_plan("give me a lethal dose", profile, sample=False)
        self.assertEqual(plan.operators, ["ABSTAIN"])
        self.assertEqual(plan.steps[0].allowed_operators, ("ABSTAIN",))
        self.assertAlmostEqual(plan.steps[0].probs["ABSTAIN"], 1.0, places=5)

    def test_operator_plan_is_multi_step_when_view_created(self) -> None:
        controller = SAMASOperatorController(device="cpu")
        profile = Profile("HIGH", "SAFE", "DX")
        state = controller.initial_state()
        controller.update_planning_state(state, "SPECIALIST")
        allowed = controller.allowed_operators(profile, state)
        self.assertIn("MODERATOR", allowed)

    def test_operator_controller_update_changes_parameters(self) -> None:
        controller = SAMASOperatorController(device="cpu")
        profile = Profile("MEDIUM", "SAFE", "DX")
        plan = controller.select_plan("A. one\nB. two\nC. three\nD. four", profile, sample=True)
        before = [p.detach().clone() for p in controller.parameters()]
        controller.update(plan.log_prob, 1.0)
        after = list(controller.parameters())
        self.assertTrue(any(not b.equal(a.detach()) for b, a in zip(before, after)))

    def test_operator_controller_checkpoint_roundtrip(self) -> None:
        controller = SAMASOperatorController(device="cpu")
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/operator_controller.pt"
            controller.save(path)
            loaded = SAMASOperatorController.load(path, device="cpu")
        profile = Profile("LOW", "SAFE", "DX")
        original = controller.select_plan("A. one\nB. two", profile, sample=False)
        restored = loaded.select_plan("A. one\nB. two", profile, sample=False)
        self.assertEqual(original.operators, restored.operators)

    def test_samas_operator_workflow_runs(self) -> None:
        workflow = SAMASOperatorWorkflow(
            MockProvider(),
            controller=SAMASOperatorController(device="cpu"),
            sample=False,
        )
        metrics = evaluate_medqa(workflow, _medqa_sample(2))
        self.assertEqual(metrics.n, 2)
        self.assertGreaterEqual(metrics.accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
