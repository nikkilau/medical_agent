"""Medical Test-Time Compute framework built on top of Micro-MDT.

Submodules:
    workflow      — Workflow base class + WorkflowResult
    operators     — Reusable Operator components (Triage/Generate/Review/...)
    baselines     — W0-W3 baseline workflows
    ttc_workflow  — W4 Medical TTC main workflow
    data          — MedQA + MedSafetyBench loaders
    evaluator     — G(W,T) = accuracy − λ·unsafe_rate
    router        — Difficulty router (zero-shot prompt or trained classifier)
    safety        — Hard safety constraint wrapper
    mcts          — AFLOW-style MCTS optimization loop
"""

from .workflow import Workflow, WorkflowResult
from .operators import (
    TriageOperator,
    GenerateOperator,
    ReviewOperator,
    ReviseOperator,
    EnsembleOperator,
    ModeratorOperator,
    CustomOperator,
)

__all__ = [
    "Workflow",
    "WorkflowResult",
    "TriageOperator",
    "GenerateOperator",
    "ReviewOperator",
    "ReviseOperator",
    "EnsembleOperator",
    "ModeratorOperator",
    "CustomOperator",
]
