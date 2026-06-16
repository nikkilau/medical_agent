"""Smoke tests for the Medical TTC framework using the deterministic
MockProvider — no network or API key required.

Run from the project root:
    PYTHONPATH=src python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from medical_ttc.baselines import (
    BaselineIOWorkflow,
    CoTSCWorkflow,
    HandCodedMicroMDTWorkflow,
    extract_mcq_letter,
)
from medical_ttc.data import _medqa_sample, _medsafety_sample, LabeledCase
from medical_ttc.evaluator import (
    compute_g_score,
    evaluate_medqa,
    evaluate_medsafety,
)
from medical_ttc.router import HeuristicRouter, train_classifier_router
from medical_ttc.source_baselines import MedAgentsSourceWorkflow, MDAgentsSourceWorkflow
from medical_ttc.ttc_workflow import MedicalTTCWorkflow
from medical_ttc.workflow import WorkflowResult
from micro_mdt.models import Difficulty, PatientCase
from micro_mdt.providers import MockProvider


class TestExtractor(unittest.TestCase):
    def test_letter_extraction(self) -> None:
        self.assertEqual(extract_mcq_letter("Final answer: B"), "B")
        self.assertEqual(extract_mcq_letter("The correct option is (C)."), "C")
        self.assertEqual(extract_mcq_letter("A. yes\nB. no\nAnswer: A"), "A")
        self.assertIsNone(extract_mcq_letter(""))
        self.assertIsNone(extract_mcq_letter("nothing here"))


class TestData(unittest.TestCase):
    def test_medqa_sample(self) -> None:
        cases = _medqa_sample(3)
        self.assertEqual(len(cases), 3)
        for lc in cases:
            self.assertIsInstance(lc, LabeledCase)
            self.assertIn(lc.ground_truth, {"A", "B", "C", "D"})
            self.assertIn("A.", lc.case.text)

    def test_medsafety_sample(self) -> None:
        cases = _medsafety_sample(3)
        self.assertEqual(len(cases), 3)
        for lc in cases:
            self.assertEqual(lc.ground_truth, "ABSTAIN")


class TestRouters(unittest.TestCase):
    def test_heuristic_router(self) -> None:
        router = HeuristicRouter()
        low = router.route("Patient has mild sore throat for 1 day, otherwise healthy.")
        self.assertEqual(low, Difficulty.LOW)
        high = router.route(
            "70 y/o on warfarin for atrial fibrillation, INR 4.5, melena, chest pain."
        )
        self.assertEqual(high, Difficulty.HIGH)

    def test_train_classifier_router(self) -> None:
        examples = [
            ("mild cold for 1 day", Difficulty.LOW),
            ("routine follow-up of hypertension", Difficulty.MEDIUM),
            ("warfarin INR 5 melena chest pain", Difficulty.HIGH),
            ("just sore throat", Difficulty.LOW),
            ("diabetes follow-up", Difficulty.MEDIUM),
            ("STEMI chest pain shortness of breath", Difficulty.HIGH),
        ]
        router = train_classifier_router(examples * 5, n_epochs=10)
        # smoke: every text routes to something
        for t, _ in examples:
            d = router.route(t)
            self.assertIn(d, list(Difficulty))


class TestBaselines(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MockProvider()
        self.cases = _medqa_sample(3)

    def test_w0_runs(self) -> None:
        wf = BaselineIOWorkflow(self.provider)
        out = wf(self.cases[0].case)
        self.assertIsInstance(out, WorkflowResult)
        self.assertGreater(len(out.final_answer), 0)

    def test_w1_runs(self) -> None:
        wf = CoTSCWorkflow(self.provider, n_samples=3)
        out = wf(self.cases[0].case)
        self.assertEqual(out.rounds, 3)

    def test_w2_runs(self) -> None:
        wf = HandCodedMicroMDTWorkflow(self.provider)
        out = wf(self.cases[0].case)
        self.assertIsInstance(out, WorkflowResult)

    def test_source_wrappers_run(self) -> None:
        medagents = MedAgentsSourceWorkflow(self.provider, method="base_direct")
        mdagents = MDAgentsSourceWorkflow(self.provider, difficulty="basic")
        self.assertEqual(medagents(self.cases[0].case).metadata["source"], "MedAgents")
        self.assertEqual(mdagents(self.cases[0].case).metadata["source"], "MDAgents")


class TestTTCWorkflow(unittest.TestCase):
    def test_w4_runs_three_tiers(self) -> None:
        provider = MockProvider()
        wf = MedicalTTCWorkflow(provider, router=HeuristicRouter())
        cases = _medqa_sample(3)
        for lc in cases:
            out = wf(lc.case)
            self.assertIsInstance(out, WorkflowResult)
            self.assertIn(out.difficulty, ["LOW", "MEDIUM", "HIGH"])

    def test_w4_no_safety(self) -> None:
        provider = MockProvider()
        wf = MedicalTTCWorkflow(provider, router=HeuristicRouter(), enable_safety=False)
        cases = _medqa_sample(2)
        for lc in cases:
            out = wf(lc.case)
            # Without safety, should never abstain (unless inner mock returns ABSTAIN)
            self.assertIsInstance(out, WorkflowResult)


class TestEvaluator(unittest.TestCase):
    def test_g_score_pipeline(self) -> None:
        provider = MockProvider()
        wf = BaselineIOWorkflow(provider)
        cases = _medqa_sample(3)
        safety_cases = _medsafety_sample(2)
        med_m = evaluate_medqa(wf, cases)
        self.assertEqual(med_m.n, 3)
        self.assertGreaterEqual(med_m.accuracy, 0.0)
        self.assertLessEqual(med_m.accuracy, 1.0)
        safety_m = evaluate_medsafety(wf, safety_cases)
        self.assertEqual(safety_m["n"], 2)
        g = compute_g_score(med_m, safety_m)
        self.assertIsInstance(g, float)


if __name__ == "__main__":
    unittest.main()
