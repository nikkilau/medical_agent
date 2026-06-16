import unittest
from collections import Counter
from pathlib import Path

from micro_mdt.case_io import load_cases
from micro_mdt.models import Difficulty


CASES_PATH = Path(__file__).resolve().parents[1] / "examples" / "cases.json"


class CaseDatasetTests(unittest.TestCase):
    def test_case_dataset_has_balanced_30_case_split(self):
        cases = load_cases(CASES_PATH)

        self.assertEqual(len(cases), 30)
        counts = Counter(case.expected_difficulty for case in cases)
        self.assertEqual(counts[Difficulty.LOW], 10)
        self.assertEqual(counts[Difficulty.MEDIUM], 10)
        self.assertEqual(counts[Difficulty.HIGH], 10)

    def test_case_ids_are_unique_and_difficulty_is_set(self):
        cases = load_cases(CASES_PATH)
        case_ids = [case.case_id for case in cases]

        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertTrue(all(case.expected_difficulty in Difficulty for case in cases))
        self.assertTrue(all(case.title.strip() for case in cases))
        self.assertTrue(all(case.text.strip() for case in cases))


if __name__ == "__main__":
    unittest.main()
