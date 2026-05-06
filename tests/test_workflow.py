import unittest

from micro_mdt.models import Difficulty, PatientCase
from micro_mdt.providers import MockProvider
from micro_mdt.visualization import render_html, render_summary_html
from micro_mdt.workflow import MicroMDT


class MicroMDTWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = MicroMDT(MockProvider(), max_rounds=3)

    def test_low_case_uses_low_compute_path(self):
        case = PatientCase(
            case_id="low",
            title="low",
            text=(
                "患者 28 岁，流涕、咽痛 1 天，"
                "低热，无胸痛、无呼吸困难。"
            ),
        )

        result = self.workflow.run_case(case)

        self.assertEqual(result.difficulty, Difficulty.LOW)
        self.assertEqual(result.status, "completed_low_compute")
        self.assertFalse(result.human_required)

    def test_anticoagulant_with_black_stool_abstains(self):
        case = PatientCase(
            case_id="high",
            title="high",
            text=(
                "患者长期服用华法林，发热头痛，"
                "家属想自行加用止痛药，"
                "今天出现黑便。"
            ),
        )

        result = self.workflow.run_case(case)

        self.assertEqual(result.difficulty, Difficulty.HIGH)
        self.assertTrue(result.human_required)
        self.assertIn("拒绝提供最终诊疗方案", result.final_answer)

    def test_human_decision_generates_documentation(self):
        decision = "安排急诊评估。"
        case = PatientCase(
            case_id="urgent",
            title="urgent",
            text=(
                "患者 61 岁，突发胸痛 40 分钟，"
                "伴出汗和呼吸困难。"
            ),
        )

        result = self.workflow.run_case(case, human_decision_callback=lambda _: decision)

        self.assertEqual(result.status, "human_decision_documented")
        self.assertEqual(result.human_decision, decision)
        self.assertIn("followup_draft", result.documents)
        self.assertIn("patient_note", result.documents)
        self.assertIn("reminder", result.documents)

    def test_high_case_generates_structured_options(self):
        case = PatientCase(
            case_id="poly_high",
            title="polypharmacy conflict",
            text=(
                "患者 71 岁，糖尿病+高血压+冠心病支架术后+痛风，"
                "服用阿司匹林+氯吡格雷+多种降糖降压药，"
                "近3天自行加用吲哚美辛，今晨黑便一次。"
            ),
        )

        result = self.workflow.run_case(case)

        self.assertEqual(result.difficulty, Difficulty.HIGH)
        self.assertTrue(result.human_required)
        self.assertIsNotNone(result.human_options)
        self.assertIn("[A]", result.human_options)
        self.assertIn("[B]", result.human_options)

    def test_medium_chronic_case_attempts_revision(self):
        case = PatientCase(
            case_id="htn_mild",
            title="hypertension mild abnormal",
            text=(
                "患者 56 岁，高血压复诊，血压 160/95 mmHg，"
                "近期睡眠不佳，偶有晨起头痛，"
                "当前口服硝苯地平+厄贝沙坦。"
            ),
        )

        result = self.workflow.run_case(case)

        self.assertEqual(result.difficulty, Difficulty.MEDIUM)
        self.assertGreaterEqual(len(result.rounds), 1)

    def test_low_simple_case_direct_output(self):
        case = PatientCase(
            case_id="simple_cold",
            title="mild cold",
            text=(
                "患者 25 岁，流涕、咽痛 1 天，"
                "体温 37.5 摄氏度，无胸痛、无呼吸困难，"
                "无慢病史，无特殊用药。"
            ),
        )

        result = self.workflow.run_case(case)

        self.assertEqual(result.difficulty, Difficulty.LOW)
        self.assertEqual(result.status, "completed_low_compute")
        self.assertEqual(len(result.rounds), 0)
        self.assertFalse(result.human_required)


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.workflow = MicroMDT(MockProvider(), max_rounds=3)

    def test_render_html_contains_expected_sections(self):
        case = PatientCase(
            case_id="test_viz",
            title="visualization test",
            text="患者 28 岁，流涕、咽痛 1 天。",
        )
        result = self.workflow.run_case(case)
        html = render_html(result)

        self.assertIn("<html", html)
        self.assertIn("test_viz", html)
        self.assertIn("工作流路由", html)
        self.assertIn("病例原文", html)
        self.assertIn("最终输出", html)

    def test_render_html_with_high_case_has_debate_and_human(self):
        case = PatientCase(
            case_id="test_high",
            title="high case",
            text="患者长期服用华法林，发热头痛，家属想自行加用止痛药，今天出现黑便。",
        )
        result = self.workflow.run_case(
            case, human_decision_callback=lambda _: "安排急诊评估。"
        )
        html = render_html(result)

        self.assertIn("辩论记录", html)
        self.assertIn("人类医生接管", html)
        self.assertIn("生成的医疗文书", html)
        self.assertIn("复诊记录草稿", html)

    def test_render_summary_html_multiple_cases(self):
        cases = [
            PatientCase("c1", "case one", "患者流涕咽痛。"),
            PatientCase("c2", "case two", "患者长期华法林，黑便。"),
        ]
        results = [self.workflow.run_case(c) for c in cases]
        html = render_summary_html(results)

        self.assertIn("<html", html)
        self.assertIn("c1", html)
        self.assertIn("c2", html)
        self.assertIn("总计 2", html)


if __name__ == "__main__":
    unittest.main()
