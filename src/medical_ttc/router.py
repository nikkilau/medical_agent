"""Difficulty Router — MaAS-inspired query-adaptive routing.

Two implementations:
    1. ZeroShotRouter — uses TriageOperator (existing Triage prompt) as a
       zero-shot LOW/MED/HIGH classifier. Default for Phase 1 pilots.
    2. ClassifierRouter — lightweight trained classifier (token length +
       specialty keywords + optional LLM embedding) over MedQA. Used in
       Phase 2+.

Both expose `.route(case_text) -> Difficulty` so workflows are agnostic.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from micro_mdt.models import Difficulty
from micro_mdt.providers import LLMProvider
from .operators import TriageOperator


class ZeroShotRouter:
    """Wraps TriageOperator. No training data required."""

    def __init__(self, provider: LLMProvider, prompt: str | None = None) -> None:
        self.triage = TriageOperator(provider, prompt=prompt)

    def route(self, case_text: str) -> Difficulty:
        diff, _ = self.triage(case_text)
        return diff


# ---------------------- lightweight feature classifier --------------

HIGH_RISK_KEYWORDS = [
    "warfarin", "华法林", "anticoagulant", "INR",
    "stroke", "卒中", "脑梗", "出血",
    "MI", "infarct", "心梗", "ACS", "STEMI",
    "sepsis", "脓毒症",
    "anaphylaxis", "过敏休克",
    "pregnan", "妊娠", "孕妇",
    "eGFR", "肌酐", "肾功能",
    "ICU", "重症",
    "child", "婴儿", "新生儿", "儿童",
    "elderly", "老人", "高龄",
    "drug interaction", "药物相互作用",
    "急症", "急性", "急诊",
]

MED_RISK_KEYWORDS = [
    "diabetes", "糖尿病", "hypertension", "高血压",
    "复诊", "follow-up",
    "调整用药", "adjust dose",
    "化验异常", "lab abnormal",
]


_ASCII_ONLY = re.compile(r"^[A-Za-z0-9_\-]+$")


def _kw_match(kw: str, text_lower: str) -> bool:
    """Word-boundary aware keyword match.

    Avoids spurious hits like "MI" matching the substring inside "mild" or
    "Cmi". For purely Latin-script keywords we use \\b boundaries; for
    Chinese keywords (no word boundaries in CJK) we do plain substring.
    """
    kw_lc = kw.lower()
    if _ASCII_ONLY.match(kw_lc):
        return re.search(r"\b" + re.escape(kw_lc) + r"\b", text_lower) is not None
    return kw_lc in text_lower


@dataclass
class HeuristicRouter:
    """Pure rule-based router used as a strong, cheap baseline.

    Maps token length + risk-keyword counts to LOW/MED/HIGH. No training.
    """

    high_threshold: int = 2  # number of high-risk keyword hits
    long_token_threshold: int = 400  # case length above which LOW is unlikely

    def route(self, case_text: str) -> Difficulty:
        text_lower = case_text.lower()
        high_hits = sum(1 for kw in HIGH_RISK_KEYWORDS if _kw_match(kw, text_lower))
        med_hits = sum(1 for kw in MED_RISK_KEYWORDS if _kw_match(kw, text_lower))
        n_tokens = len(re.findall(r"\S+", case_text))
        if high_hits >= self.high_threshold or n_tokens > self.long_token_threshold:
            return Difficulty.HIGH
        if high_hits >= 1 or med_hits >= 1:
            return Difficulty.MEDIUM
        return Difficulty.LOW


# ---------------------- trained classifier --------------------------


@dataclass
class _LinearClassifier:
    """Tiny linear scorer over a feature vector."""

    weights: dict[str, float]
    bias: dict[str, float]

    def score(self, feats: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for cls in ("LOW", "MEDIUM", "HIGH"):
            s = self.bias.get(cls, 0.0)
            for k, v in feats.items():
                s += self.weights.get(f"{cls}::{k}", 0.0) * v
            scores[cls] = s
        return scores


def _featurize(text: str) -> dict[str, float]:
    text_lower = text.lower()
    n_tokens = len(re.findall(r"\S+", text))
    high_hits = sum(1 for kw in HIGH_RISK_KEYWORDS if _kw_match(kw, text_lower))
    med_hits = sum(1 for kw in MED_RISK_KEYWORDS if _kw_match(kw, text_lower))
    return {
        "n_tokens_log": (n_tokens + 1) ** 0.5,
        "high_kw": float(high_hits),
        "med_kw": float(med_hits),
        "has_number": float(bool(re.search(r"\d", text))),
        "has_dose": float(bool(re.search(r"\bmg\b|\bmcg\b|\bml\b|\bU\b", text_lower))),
    }


class ClassifierRouter:
    """MaAS-style learned router.

    Stores a small linear model. Trained by `train_classifier_router(...)`
    from labeled MedQA examples.
    """

    def __init__(self, model: _LinearClassifier) -> None:
        self.model = model

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ClassifierRouter":
        data = json.loads(Path(path).read_text())
        return cls(_LinearClassifier(weights=data["weights"], bias=data["bias"]))

    def save(self, path: str | os.PathLike) -> None:
        Path(path).write_text(
            json.dumps({"weights": self.model.weights, "bias": self.model.bias}, indent=2)
        )

    def route(self, case_text: str) -> Difficulty:
        feats = _featurize(case_text)
        scores = self.model.score(feats)
        best = max(scores, key=scores.get)
        return Difficulty[best]


def train_classifier_router(
    labeled_examples: list[tuple[str, Difficulty]],
    *,
    n_epochs: int = 20,
    lr: float = 0.05,
    seed: int = 42,
) -> ClassifierRouter:
    """Train a tiny linear classifier on (text, difficulty) pairs.

    Uses a simple perceptron-style update — adequate for ~500 examples and
    the 5-feature vector. No gradient libs required.
    """
    import random as _random

    classes = ["LOW", "MEDIUM", "HIGH"]
    rng = _random.Random(seed)
    # Initialize zero weights
    weights: dict[str, float] = {}
    bias: dict[str, float] = {c: 0.0 for c in classes}
    feat_keys = list(_featurize("").keys())
    for c in classes:
        for k in feat_keys:
            weights[f"{c}::{k}"] = 0.0
    pairs = list(labeled_examples)
    for _ in range(n_epochs):
        rng.shuffle(pairs)
        for text, diff in pairs:
            feats = _featurize(text)
            target = diff.value
            model = _LinearClassifier(weights, bias)
            scores = model.score(feats)
            pred = max(scores, key=scores.get)
            if pred == target:
                continue
            # Increase target weights, decrease predicted-wrong weights
            for k, v in feats.items():
                weights[f"{target}::{k}"] = weights.get(f"{target}::{k}", 0.0) + lr * v
                weights[f"{pred}::{k}"] = weights.get(f"{pred}::{k}", 0.0) - lr * v
            bias[target] = bias.get(target, 0.0) + lr
            bias[pred] = bias.get(pred, 0.0) - lr
    return ClassifierRouter(_LinearClassifier(weights, bias))
