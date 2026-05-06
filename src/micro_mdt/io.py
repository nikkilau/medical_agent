from __future__ import annotations

import json
from pathlib import Path

from .models import Difficulty, PatientCase


def load_cases(path: Path) -> list[PatientCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for item in data:
        expected = item.get("expected_difficulty")
        cases.append(
            PatientCase(
                case_id=item["case_id"],
                title=item["title"],
                text=item["text"],
                expected_difficulty=Difficulty(expected) if expected else None,
                metadata=item.get("metadata", {}),
            )
        )
    return cases


def load_case_from_text(text: str) -> PatientCase:
    return PatientCase(case_id="ad_hoc", title="临时病例", text=text)

