from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Difficulty(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Verdict(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class PatientCase:
    case_id: str
    title: str
    text: str
    expected_difficulty: Difficulty | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass
class AgentResponse:
    agent: str
    content: str
    verdict: Verdict | None = None


@dataclass
class DebateRound:
    round_index: int
    proposal: AgentResponse
    pharmacy_review: AgentResponse
    safety_review: AgentResponse


@dataclass
class MDTResult:
    case: PatientCase
    difficulty: Difficulty
    status: str
    final_answer: str
    rounds: list[DebateRound] = field(default_factory=list)
    human_required: bool = False
    human_options: str | None = None
    human_decision: str | None = None
    documents: dict[str, str] = field(default_factory=dict)
    trace: list[AgentResponse] = field(default_factory=list)

