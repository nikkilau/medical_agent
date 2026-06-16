from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .profiler import MedicalProfile

RouteName = Literal[
    "ACCURACY",
    "DYNAMIC_MDT",
    "SAFETY_REVIEW_LOW",
    "SAFETY_REVIEW_HIGH",
]


@dataclass(frozen=True)
class TTCRoute:
    name: RouteName
    reason: str


class DifficultySafetyRouter:
    """Four-quadrant router: safety risk first, difficulty controls compute."""

    def route(self, profile: MedicalProfile) -> TTCRoute:
        if profile.safety_risk == "HIGH":
            if profile.difficulty == "COMPLEX":
                return TTCRoute("SAFETY_REVIEW_HIGH", "high safety risk + complex")
            return TTCRoute("SAFETY_REVIEW_LOW", "high safety risk + simple")
        if profile.difficulty == "COMPLEX":
            return TTCRoute("DYNAMIC_MDT", "low safety risk + complex")
        return TTCRoute("ACCURACY", "low safety risk + simple")
