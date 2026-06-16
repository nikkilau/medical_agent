from __future__ import annotations

import math
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .profiler import Profile


TEMPLATE_IDS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10")
COMPLEXITIES = ("LOW", "MEDIUM", "HIGH")
RISKS = ("SAFE", "MODERATE", "HIGH_RISK")
TASK_TYPES = ("DX", "RX", "TRIAGE", "EDU", "ETHICS", "EVIDENCE")
OPERATOR_IDS = (
    "LOOKUP",
    "GENERALIST",
    "SPECIALIST",
    "VOTER",
    "MODERATOR",
    "PHARMACY",
    "SAFETY",
    "REVISER",
    "DIRECT",
    "ABSTAIN",
    "STOP",
)

OPERATOR_DESCRIPTIONS = {
    "LOOKUP": "Retrieve compact medical safety context such as red flags, medication interactions, contraindications, and benchmark-specific clues.",
    "GENERALIST": "Generate a broad conservative medical reasoning view for the question.",
    "SPECIALIST": "Generate a domain-specialist medical opinion, adapting to diagnosis, pharmacology, emergency triage, or evidence tasks.",
    "VOTER": "Run multiple candidate answers and use self-consistency voting to select the most likely option.",
    "MODERATOR": "Synthesize multiple views or a draft into one final benchmark answer.",
    "PHARMACY": "Audit medication safety, interactions, dosing, pregnancy, renal impairment, and contraindications.",
    "SAFETY": "Audit emergency delays, harmful instructions, overreach, and safe refusal requirements.",
    "REVISER": "Revise a draft answer using pharmacy or safety feedback.",
    "DIRECT": "Answer the medical multiple-choice question directly with concise reasoning.",
    "ABSTAIN": "Refuse or hand off unsafe medical requests instead of giving dangerous advice.",
    "STOP": "Stop architecture execution and return the current draft.",
}


@dataclass
class TemplateSelection:
    template_id: str
    reason: str
    log_prob: torch.Tensor | None = None
    probs: dict[str, float] | None = None
    allowed_templates: tuple[str, ...] = TEMPLATE_IDS


@dataclass
class OperatorStepSelection:
    operator_id: str
    step: int
    log_prob: torch.Tensor | None
    probs: dict[str, float]
    allowed_operators: tuple[str, ...]
    layer: int = 0


@dataclass
class OperatorPlanSelection:
    operators: list[str]
    steps: list[OperatorStepSelection]
    log_prob: torch.Tensor | None
    layers: list[list[str]] | None = None

    @property
    def probs_trace(self) -> list[dict[str, float]]:
        return [step.probs for step in self.steps]


class SAMASTemplateController(torch.nn.Module):
    """MaAS-style learned controller for SAMAS architecture templates.

    SAMAS templates A1-A10 are complete workflows, so this controller samples
    exactly one template per query. It reuses MaAS' core training idea: a small
    neural policy trained by REINFORCE from workflow reward.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        lr: float = 1e-2,
        prior_strength: float = 2.0,
        entropy_coef: float = 0.0,
        device: str | torch.device | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.prior_strength = prior_strength
        self.entropy_coef = entropy_coef
        self.input_dim = len(COMPLEXITIES) + len(RISKS) + len(TASK_TYPES) + 8
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, len(TEMPLATE_IDS)),
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.reward_baseline: float | None = None

    def forward(self, query: str, profile: Profile) -> torch.Tensor:
        features = self.encode_features(query, profile).to(self.device)
        logits = self.net(features)
        logits = logits + self._prior_logits(profile).to(self.device)
        return self.apply_safety_mask(logits, profile)

    def select_template(
        self,
        query: str,
        profile: Profile,
        *,
        sample: bool = True,
    ) -> TemplateSelection:
        logits = self.forward(query, profile)
        probs_tensor = F.softmax(logits, dim=-1)
        allowed = self.allowed_templates(profile)
        if sample:
            dist = torch.distributions.Categorical(probs=probs_tensor)
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
        else:
            idx = torch.argmax(probs_tensor)
            log_prob = torch.log(probs_tensor[idx].clamp_min(1e-12))
        template_id = TEMPLATE_IDS[int(idx.item())]
        probs = {
            template: float(probs_tensor[i].detach().cpu().item())
            for i, template in enumerate(TEMPLATE_IDS)
        }
        return TemplateSelection(
            template_id=template_id,
            reason="learned MaAS-style template policy",
            log_prob=log_prob,
            probs=probs,
            allowed_templates=allowed,
        )

    def update(self, log_prob: torch.Tensor, reward: float) -> dict[str, float]:
        if not torch.is_tensor(log_prob):
            raise TypeError("log_prob must be the tensor returned by select_template(sample=True)")
        reward_value = float(reward)
        baseline = 0.0 if self.reward_baseline is None else self.reward_baseline
        advantage = reward_value - baseline

        self.optimizer.zero_grad()
        loss = -log_prob * advantage
        if self.entropy_coef:
            loss = loss - self.entropy_coef * (-log_prob)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optimizer.step()
        if self.reward_baseline is None:
            self.reward_baseline = reward_value
        else:
            self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward_value
        return {
            "loss": float(loss.detach().cpu().item()),
            "reward": reward_value,
            "baseline": float(self.reward_baseline),
            "advantage": float(advantage),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "reward_baseline": self.reward_baseline,
                "config": {
                    "hidden_dim": self.hidden_dim,
                    "lr": self.lr,
                    "prior_strength": self.prior_strength,
                    "entropy_coef": self.entropy_coef,
                    "input_dim": self.input_dim,
                    "template_ids": TEMPLATE_IDS,
                },
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device | None = None) -> "SAMASTemplateController":
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg = ckpt.get("config", {})
        controller = cls(
            hidden_dim=int(cfg.get("hidden_dim", 32)),
            lr=float(cfg.get("lr", 1e-2)),
            prior_strength=float(cfg.get("prior_strength", 2.0)),
            entropy_coef=float(cfg.get("entropy_coef", 0.0)),
            device=device,
        )
        controller.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            controller.optimizer.load_state_dict(ckpt["optimizer"])
        controller.reward_baseline = ckpt.get("reward_baseline")
        return controller

    @staticmethod
    def encode_features(query: str, profile: Profile) -> torch.Tensor:
        lower = query.lower()
        tokens = re.findall(r"\w+", lower)
        option_count = len(re.findall(r"(?m)^\s*[A-E][\).\s]", query))
        flags = [
            min(len(tokens) / 256.0, 2.0),
            min(len(query) / 2048.0, 2.0),
            _has_any(lower, _DRUG_TERMS),
            _has_any(lower, _EMERGENCY_TERMS),
            _has_any(lower, _HARMFUL_TERMS),
            _has_any(lower, _PREGNANCY_PEDIATRIC_TERMS),
            1.0 if option_count >= 2 or "a/b/c/d" in lower else 0.0,
            min(option_count / 5.0, 1.0),
        ]
        values: list[float] = []
        values.extend(_one_hot(profile.complexity, COMPLEXITIES))
        values.extend(_one_hot(profile.risk, RISKS))
        values.extend(_one_hot(profile.task_type, TASK_TYPES))
        values.extend(flags)
        return torch.tensor(values, dtype=torch.float32)

    def _prior_logits(self, profile: Profile) -> torch.Tensor:
        logits = torch.zeros(len(TEMPLATE_IDS), dtype=torch.float32)
        preferred = rule_prior_template(profile)
        logits[TEMPLATE_IDS.index(preferred)] = self.prior_strength
        if profile.task_type == "RX":
            logits[TEMPLATE_IDS.index("A7")] += self.prior_strength * 0.75
        if profile.task_type == "TRIAGE":
            logits[TEMPLATE_IDS.index("A9")] += self.prior_strength * 0.75
        if profile.risk != "SAFE":
            for template in ("A3", "A5", "A6"):
                logits[TEMPLATE_IDS.index(template)] += self.prior_strength * 0.25
        return logits

    def apply_safety_mask(self, logits: torch.Tensor, profile: Profile) -> torch.Tensor:
        allowed = set(self.allowed_templates(profile))
        mask = torch.full_like(logits, -math.inf)
        for idx, template in enumerate(TEMPLATE_IDS):
            if template in allowed:
                mask[idx] = 0.0
        return logits + mask

    @staticmethod
    def allowed_templates(profile: Profile) -> tuple[str, ...]:
        if profile.task_type == "ETHICS":
            return ("A10",)
        if profile.task_type == "TRIAGE":
            return ("A3", "A5", "A6", "A9")
        if profile.task_type == "RX":
            return ("A3", "A5", "A6", "A7")
        if profile.task_type == "EDU":
            return ("A3", "A5", "A8")
        if profile.risk == "HIGH_RISK":
            return ("A3", "A5", "A6", "A9", "A10")
        if profile.risk == "MODERATE":
            return ("A3", "A4", "A5", "A6", "A7", "A8", "A9")
        return TEMPLATE_IDS


class _SAMASOperatorSelector(torch.nn.Module):
    """One MaAS controller layer: query/state embedding scores operator embeddings."""

    def __init__(
        self,
        *,
        query_dim: int,
        operator_dim: int,
        hidden_dim: int,
        is_first_layer: bool,
    ) -> None:
        super().__init__()
        self.is_first_layer = is_first_layer
        self.query_encoder = torch.nn.Linear(query_dim, hidden_dim)
        self.operator_encoder = torch.nn.Linear(
            operator_dim if is_first_layer else operator_dim * 2,
            hidden_dim,
        )

    def forward(
        self,
        query_features: torch.Tensor,
        operator_embeddings: torch.Tensor,
        prev_operator_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        query_hidden = F.normalize(self.query_encoder(query_features), p=2, dim=1)

        if prev_operator_embedding is not None and not self.is_first_layer:
            if prev_operator_embedding.dim() == 1:
                prev_operator_embedding = prev_operator_embedding.unsqueeze(0)
            prev = prev_operator_embedding.expand(operator_embeddings.size(0), -1)
            op_input = torch.cat([operator_embeddings, prev], dim=1)
        else:
            op_input = operator_embeddings

        operator_hidden = F.normalize(self.operator_encoder(op_input), p=2, dim=1)
        return torch.matmul(query_hidden, operator_hidden.T).squeeze(0)


class SAMASOperatorController(torch.nn.Module):
    """MaAS-style multilayer controller over medical operators.

    This mirrors the MaAS source design more closely than template routing:
    every layer scores all candidate operators by query/state embeddings and
    operator-description embeddings, samples one or more operators until a
    probability-mass threshold is reached, executes them, then feeds the
    selected operator embedding into the next layer.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 48,
        lr: float = 1e-2,
        max_steps: int = 6,
        semantic_dim: int = 32,
        operator_embedding_dim: int | None = None,
        selection_threshold: float = 0.3,
        advantage_clip: float = 5.0,
        entropy_coef: float = 0.0,
        device: str | torch.device | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.max_steps = max_steps
        self.semantic_dim = semantic_dim
        self.operator_embedding_dim = operator_embedding_dim or semantic_dim
        self.selection_threshold = selection_threshold
        self.advantage_clip = advantage_clip
        self.entropy_coef = entropy_coef
        self.base_input_dim = len(COMPLEXITIES) + len(RISKS) + len(TASK_TYPES) + 8
        self.state_dim = 8 + len(OPERATOR_IDS) + semantic_dim
        self.input_dim = self.base_input_dim + self.state_dim
        self.layers = torch.nn.ModuleList(
            [
                _SAMASOperatorSelector(
                    query_dim=self.input_dim,
                    operator_dim=self.operator_embedding_dim,
                    hidden_dim=hidden_dim,
                    is_first_layer=(i == 0),
                )
                for i in range(max_steps)
            ]
        )
        self.register_buffer(
            "operator_embeddings",
            torch.tensor(
                [
                    _description_embedding(OPERATOR_DESCRIPTIONS[op], self.operator_embedding_dim)
                    for op in OPERATOR_IDS
                ],
                dtype=torch.float32,
            ),
        )
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.reward_baseline: float | None = None

    def forward(self, query: str, profile: Profile, state: dict) -> torch.Tensor:
        layer_idx = min(int(state.get("layer", state.get("step", 0))), len(self.layers) - 1)
        features = self.encode_features(query, profile, state).to(self.device)
        prev_embedding = state.get("prev_operator_embedding")
        prev_tensor = None
        if isinstance(prev_embedding, list) and prev_embedding:
            prev_tensor = torch.tensor(prev_embedding, dtype=torch.float32, device=self.device)
        logits = self.layers[layer_idx](
            features,
            self.operator_embeddings.to(self.device),
            prev_tensor,
        )
        logits = logits + self._prior_logits(profile, state).to(self.device)
        return self.apply_action_mask(logits, profile, state)

    def set_operator_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        rows = []
        for op in OPERATOR_IDS:
            values = embeddings.get(op) or []
            rows.append(_fit_vector(values, self.operator_embedding_dim))
        tensor = torch.tensor(rows, dtype=torch.float32, device=self.device)
        norm = tensor.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        self.operator_embeddings.data.copy_(tensor / norm)

    def select_plan(
        self,
        query: str,
        profile: Profile,
        *,
        sample: bool = True,
    ) -> OperatorPlanSelection:
        state = self.initial_state()
        steps: list[OperatorStepSelection] = []
        log_probs: list[torch.Tensor] = []
        operators: list[str] = []

        for step_idx in range(self.max_steps):
            state["layer"] = step_idx
            layer_steps = self.select_layer(query, profile, state, layer=step_idx, sample=sample)
            layer_ops: list[str] = []
            for selected in layer_steps:
                op = selected.operator_id
                steps.append(selected)
                operators.append(op)
                layer_ops.append(op)
                if selected.log_prob is not None:
                    log_probs.append(selected.log_prob)
                self.update_planning_state(state, op)
            self._set_prev_operator_embedding(state, [s.operator_id for s in layer_steps])
            if any(op in {"STOP", "ABSTAIN"} for op in layer_ops):
                break

        total_log_prob = torch.stack(log_probs).sum() if log_probs else None
        layers = []
        for step in steps:
            while len(layers) <= step.layer:
                layers.append([])
            layers[step.layer].append(step.operator_id)
        return OperatorPlanSelection(operators=operators, steps=steps, log_prob=total_log_prob, layers=layers)

    def select_layer(
        self,
        query: str,
        profile: Profile,
        state: dict,
        *,
        layer: int | None = None,
        sample: bool = True,
    ) -> list[OperatorStepSelection]:
        layer_idx = int(state.get("step", 0) if layer is None else layer)
        logits = self.forward(query, profile, state)
        probs_tensor = F.softmax(logits, dim=-1)
        log_probs_tensor = F.log_softmax(logits, dim=-1)
        allowed = self.allowed_operators(profile, state)
        if sample:
            indices = _sample_operators_until_threshold(probs_tensor, threshold=self.selection_threshold)
        else:
            indices = _greedy_operators_until_threshold(probs_tensor, threshold=self.selection_threshold)
        probs = {
            operator: float(probs_tensor[i].detach().cpu().item())
            for i, operator in enumerate(OPERATOR_IDS)
        }
        selections: list[OperatorStepSelection] = []
        used: set[str] = set()
        for idx in indices:
            op = OPERATOR_IDS[int(idx.item())]
            if op in used:
                continue
            used.add(op)
            selections.append(
                OperatorStepSelection(
                    operator_id=op,
                    step=int(state.get("step", 0)),
                    log_prob=log_probs_tensor[idx],
                    probs=probs,
                    allowed_operators=allowed,
                    layer=layer_idx,
                )
            )
            if op in {"STOP", "ABSTAIN"}:
                break
        if not selections:
            return [self.select_next(query, profile, state, step=layer_idx, sample=False)]
        return selections

    def select_next(
        self,
        query: str,
        profile: Profile,
        state: dict,
        *,
        step: int | None = None,
        sample: bool = True,
    ) -> OperatorStepSelection:
        logits = self.forward(query, profile, state)
        probs_tensor = F.softmax(logits, dim=-1)
        allowed = self.allowed_operators(profile, state)
        if sample:
            dist = torch.distributions.Categorical(probs=probs_tensor)
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
        else:
            idx = torch.argmax(probs_tensor)
            log_prob = torch.log(probs_tensor[idx].clamp_min(1e-12))
        op = OPERATOR_IDS[int(idx.item())]
        self._set_prev_operator_embedding(state, [op])
        probs = {
            operator: float(probs_tensor[i].detach().cpu().item())
            for i, operator in enumerate(OPERATOR_IDS)
        }
        return OperatorStepSelection(
            operator_id=op,
            step=int(state.get("step", 0) if step is None else step),
            log_prob=log_prob,
            probs=probs,
            allowed_operators=allowed,
            layer=int(state.get("step", 0) if step is None else step),
        )

    def update(self, log_prob: torch.Tensor, reward: float) -> dict[str, float]:
        if not torch.is_tensor(log_prob):
            raise TypeError("log_prob must be returned by select_plan(sample=True)")
        reward_value = float(reward)
        baseline = 0.0 if self.reward_baseline is None else self.reward_baseline
        advantage = reward_value - baseline
        unclipped_advantage = advantage
        if self.advantage_clip > 0:
            advantage = max(-self.advantage_clip, min(self.advantage_clip, advantage))

        self.optimizer.zero_grad()
        loss = -log_prob * advantage
        if self.entropy_coef:
            loss = loss - self.entropy_coef * (-log_prob)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optimizer.step()
        if self.reward_baseline is None:
            self.reward_baseline = reward_value
        else:
            self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward_value
        return {
            "loss": float(loss.detach().cpu().item()),
            "reward": reward_value,
            "baseline": float(self.reward_baseline),
            "advantage": float(advantage),
            "unclipped_advantage": float(unclipped_advantage),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "reward_baseline": self.reward_baseline,
                "config": {
                    "hidden_dim": self.hidden_dim,
                    "lr": self.lr,
                    "max_steps": self.max_steps,
                    "semantic_dim": self.semantic_dim,
                    "operator_embedding_dim": self.operator_embedding_dim,
                    "selection_threshold": self.selection_threshold,
                    "advantage_clip": self.advantage_clip,
                    "entropy_coef": self.entropy_coef,
                    "operator_ids": OPERATOR_IDS,
                    "operator_descriptions": OPERATOR_DESCRIPTIONS,
                },
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device | None = None) -> "SAMASOperatorController":
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg = ckpt.get("config", {})
        controller = cls(
            hidden_dim=int(cfg.get("hidden_dim", 48)),
            lr=float(cfg.get("lr", 1e-2)),
            max_steps=int(cfg.get("max_steps", 6)),
            semantic_dim=int(cfg.get("semantic_dim", 32)),
            operator_embedding_dim=int(cfg.get("operator_embedding_dim", cfg.get("semantic_dim", 32))),
            selection_threshold=float(cfg.get("selection_threshold", 0.3)),
            advantage_clip=float(cfg.get("advantage_clip", 5.0)),
            entropy_coef=float(cfg.get("entropy_coef", 0.0)),
            device=device,
        )
        controller.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            controller.optimizer.load_state_dict(ckpt["optimizer"])
        controller.reward_baseline = ckpt.get("reward_baseline")
        return controller

    def initial_state(self) -> dict:
        return {
            "step": 0,
            "has_lookup": False,
            "has_draft": False,
            "n_views": 0,
            "has_pharmacy": False,
            "has_safety": False,
            "has_revised": False,
            "abstained": False,
            "semantic": [0.0] * self.semantic_dim,
            "prev_operator_embedding": [],
            "layer": 0,
            "used": set(),
        }

    @staticmethod
    def update_planning_state(state: dict, operator_id: str) -> None:
        state["step"] = int(state.get("step", 0)) + 1
        state.setdefault("used", set()).add(operator_id)
        if operator_id == "LOOKUP":
            state["has_lookup"] = True
        elif operator_id in {"DIRECT", "VOTER"}:
            state["has_draft"] = True
        elif operator_id in {"GENERALIST", "SPECIALIST"}:
            state["n_views"] = int(state.get("n_views", 0)) + 1
        elif operator_id == "MODERATOR":
            state["has_draft"] = True
        elif operator_id == "PHARMACY":
            state["has_pharmacy"] = True
        elif operator_id == "SAFETY":
            state["has_safety"] = True
        elif operator_id == "REVISER":
            state["has_revised"] = True
            state["has_draft"] = True
        elif operator_id == "ABSTAIN":
            state["abstained"] = True
            state["has_draft"] = True

    def _set_prev_operator_embedding(self, state: dict, operators: list[str]) -> None:
        indices = [OPERATOR_IDS.index(op) for op in operators if op in OPERATOR_IDS]
        if not indices:
            state["prev_operator_embedding"] = []
            return
        tensor = self.operator_embeddings[indices].detach().mean(dim=0)
        state["prev_operator_embedding"] = [float(v) for v in tensor.cpu().tolist()]

    def encode_features(self, query: str, profile: Profile, state: dict) -> torch.Tensor:
        base = SAMASTemplateController.encode_features(query, profile).tolist()
        used = state.get("used", set())
        step = float(state.get("step", 0))
        state_features = [
            min(step / 6.0, 1.0),
            1.0 if state.get("has_lookup") else 0.0,
            1.0 if state.get("has_draft") else 0.0,
            min(float(state.get("n_views", 0)) / 3.0, 1.0),
            1.0 if state.get("has_pharmacy") else 0.0,
            1.0 if state.get("has_safety") else 0.0,
            1.0 if state.get("has_revised") else 0.0,
            1.0 if state.get("abstained") else 0.0,
        ]
        state_features.extend(1.0 if op in used else 0.0 for op in OPERATOR_IDS)
        state_features.extend(self._semantic_features(state.get("semantic")))
        return torch.tensor(base + state_features, dtype=torch.float32)

    def _semantic_features(self, values: object) -> list[float]:
        if not isinstance(values, list):
            return [0.0] * self.semantic_dim
        out = [float(v) for v in values[: self.semantic_dim]]
        if len(out) < self.semantic_dim:
            out.extend([0.0] * (self.semantic_dim - len(out)))
        return out

    def _prior_logits(self, profile: Profile, state: dict) -> torch.Tensor:
        logits = torch.zeros(len(OPERATOR_IDS), dtype=torch.float32)
        if profile.task_type == "ETHICS":
            logits[OPERATOR_IDS.index("ABSTAIN")] += 4.0
        if profile.task_type == "RX" and not state.get("has_lookup"):
            logits[OPERATOR_IDS.index("LOOKUP")] += 2.0
        if profile.task_type == "TRIAGE":
            logits[OPERATOR_IDS.index("SPECIALIST")] += 1.0
            logits[OPERATOR_IDS.index("SAFETY")] += 1.0
        if profile.complexity == "HIGH":
            logits[OPERATOR_IDS.index("SPECIALIST")] += 1.0
            logits[OPERATOR_IDS.index("MODERATOR")] += 0.8
        if profile.complexity == "LOW" and profile.risk == "SAFE":
            logits[OPERATOR_IDS.index("DIRECT")] += 1.5
        if state.get("n_views", 0) > 0:
            logits[OPERATOR_IDS.index("MODERATOR")] += 1.0
        if state.get("has_draft") and self.requires_safety(profile) and not state.get("has_safety"):
            logits[OPERATOR_IDS.index("SAFETY")] += 2.0
        if state.get("has_draft") and not self.requires_safety(profile):
            logits[OPERATOR_IDS.index("STOP")] += 0.8
        return logits

    def apply_action_mask(self, logits: torch.Tensor, profile: Profile, state: dict) -> torch.Tensor:
        allowed = set(self.allowed_operators(profile, state))
        mask = torch.full_like(logits, -math.inf)
        for idx, operator in enumerate(OPERATOR_IDS):
            if operator in allowed:
                mask[idx] = 0.0
        return logits + mask

    def allowed_operators(self, profile: Profile, state: dict) -> tuple[str, ...]:
        if profile.task_type == "ETHICS":
            return ("ABSTAIN",)

        allowed: list[str] = []
        used = state.get("used", set())
        has_draft = bool(state.get("has_draft"))
        has_views = int(state.get("n_views", 0)) > 0
        needs_safety = self.requires_safety(profile)

        if not state.get("has_lookup") and profile.task_type in {"RX", "TRIAGE", "EVIDENCE"}:
            allowed.append("LOOKUP")
        if not has_draft:
            allowed.extend(["DIRECT", "GENERALIST", "SPECIALIST", "VOTER"])
        if has_views or has_draft:
            allowed.append("MODERATOR")

        if has_draft and not state.get("has_pharmacy") and profile.task_type == "RX":
            allowed.append("PHARMACY")
        if has_draft and not state.get("has_safety") and needs_safety:
            allowed.append("SAFETY")
        if has_draft and (state.get("has_pharmacy") or state.get("has_safety")) and not state.get("has_revised"):
            allowed.append("REVISER")
        if has_draft and (not needs_safety or state.get("has_safety")):
            allowed.append("STOP")
        if profile.risk == "HIGH_RISK":
            allowed.append("ABSTAIN")

        if int(state.get("step", 0)) >= self.max_steps - 1:
            allowed = ["STOP"] if has_draft else ["DIRECT"]

        # Avoid useless repeats except specialists/views and moderator after views.
        repeatable = {"SPECIALIST", "GENERALIST"}
        filtered = [op for op in allowed if op not in used or op in repeatable]
        return tuple(dict.fromkeys(filtered or ["DIRECT"]))

    @staticmethod
    def requires_safety(profile: Profile) -> bool:
        return profile.risk != "SAFE" or profile.task_type in {"RX", "TRIAGE", "EDU"}


def rule_prior_template(profile: Profile) -> str:
    if profile.task_type == "ETHICS":
        return "A10"
    if profile.task_type == "TRIAGE":
        return "A9"
    if profile.task_type == "RX":
        return "A7"
    if profile.task_type == "EDU":
        return "A8"
    if profile.complexity == "HIGH":
        return "A6"
    if profile.risk != "SAFE":
        return "A3" if profile.complexity == "LOW" else "A5"
    if profile.complexity == "MEDIUM":
        return "A4"
    return "A1"


def _one_hot(value: str, choices: Iterable[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _has_any(text: str, terms: tuple[str, ...]) -> float:
    return 1.0 if any(term in text for term in terms) else 0.0


def _sample_operators_until_threshold(probs: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """MaAS-style sample-without-replacement until cumulative mass threshold."""
    probs = probs.detach()
    remaining = torch.arange(probs.numel(), device=probs.device)
    selected: list[torch.Tensor] = []
    cumulative = 0.0
    while cumulative < threshold and remaining.numel() > 0:
        local_probs = probs[remaining]
        total = local_probs.sum()
        if float(total.item()) <= 0:
            break
        sampled_local = torch.multinomial(local_probs / total, num_samples=1)
        idx = remaining[sampled_local].squeeze()
        selected.append(idx)
        cumulative += float(probs[idx].item())
        keep = torch.ones_like(remaining, dtype=torch.bool)
        keep[sampled_local] = False
        remaining = remaining[keep]
    if not selected:
        return torch.argmax(probs).view(1)
    return torch.stack(selected)


def _greedy_operators_until_threshold(probs: torch.Tensor, *, threshold: float) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    cumulative = 0.0
    for idx in torch.argsort(probs.detach(), descending=True):
        selected.append(idx)
        cumulative += float(probs[idx].item())
        if cumulative >= threshold:
            break
    if not selected:
        return torch.argmax(probs).view(1)
    return torch.stack(selected)


def _description_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[i % len(digest)] / 127.5) - 1.0) for i in range(dim)]
    return _fit_vector(values, dim)


def _fit_vector(values: Iterable[float], dim: int) -> list[float]:
    if dim <= 0:
        return []
    out = [float(v) for v in list(values)[:dim]]
    if len(out) < dim:
        out.extend([0.0] * (dim - len(out)))
    norm = math.sqrt(sum(v * v for v in out))
    if norm > 0:
        out = [v / norm for v in out]
    return out


_DRUG_TERMS = (
    "warfarin",
    "ibuprofen",
    "aspirin",
    "metformin",
    "insulin",
    "lisinopril",
    "labetalol",
    "dose",
    "contraindication",
    "interaction",
    "药",
    "剂量",
)

_EMERGENCY_TERMS = (
    "chest pain",
    "shortness of breath",
    "diaphoresis",
    "stroke",
    "weakness",
    "seizure",
    "st elevation",
    "急",
    "胸痛",
    "呼吸困难",
)

_HARMFUL_TERMS = (
    "lethal",
    "fatal dose",
    "without telling",
    "do not call",
    "avoid the doctor",
    "致命",
    "不要告诉医生",
    "不要叫救护车",
)

_PREGNANCY_PEDIATRIC_TERMS = (
    "pregnant",
    "pregnancy",
    "gestation",
    "infant",
    "child",
    "pediatric",
    "孕",
    "婴儿",
    "儿童",
)
