import unittest

from micro_mdt.workflows.dynamic_mdt import _LocalDynamicMDT


class DynamicMDTParsingTests(unittest.TestCase):
    def test_parse_json_roles(self):
        raw = (
            '<think>reasoning that should be ignored</think>\n'
            '{"roles":["Hepatologist - liver injury",'
            '"Clinical toxicologist - drug exposure",'
            '"Clinical pharmacologist - medication review"]}'
        )

        roles = _LocalDynamicMDT._parse_roles(raw)

        self.assertEqual(len(roles), 3)
        self.assertEqual(roles[0], "Hepatologist - liver injury")
        self.assertEqual(roles[1], "Clinical toxicologist - drug exposure")
        self.assertEqual(roles[2], "Clinical pharmacologist - medication review")

    def test_thinking_answer_is_not_used_as_roles(self):
        raw = (
            "<think>long reasoning about acetaminophen toxicity</think>\n"
            "1. 对乙酰氨基酚：肝毒性，常见于过量使用。\n"
            "答案：B. 对乙酰氨基酚"
        )

        roles = _LocalDynamicMDT._parse_roles(raw)

        self.assertEqual(roles, [])

    def test_fallback_roles_for_liver_question(self):
        roles = _LocalDynamicMDT._fallback_roles(
            "57岁男性黄疸，AST 1110 U/L，ALT 990 U/L，PT/PTT延长。"
        )

        self.assertEqual(len(roles), 3)
        self.assertIn("Hepatologist", roles[0])
        self.assertIn("toxicologist", roles[1])
        self.assertIn("pharmacologist", roles[2])

    def test_unclosed_thinking_is_stripped(self):
        raw = "<think>hidden chain of thought\nFinal answer: B"

        cleaned = _LocalDynamicMDT._strip_thinking(raw)

        self.assertEqual(cleaned, "")


if __name__ == "__main__":
    unittest.main()
