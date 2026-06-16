"""Workflow abstraction — base class for W0..W7."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from micro_mdt.models import PatientCase


@dataclass
class WorkflowResult:
    """Output of running a Workflow on a single case."""

    case: PatientCase
    final_answer: str
    abstained: bool = False
    rounds: int = 0
    token_cost: float = 0.0
    difficulty: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Workflow(ABC):
    """Base class. A workflow is a function `case -> WorkflowResult`.

    Compatible with AFLOW's code-as-workflow representation: each subclass
    defines `__call__` as the executable workflow body. The MCTS optimizer
    will generate new subclasses by editing the body and prompts.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(self, case: PatientCase) -> WorkflowResult: ...

    def __repr__(self) -> str:
        return f"<Workflow {self.name}>"
