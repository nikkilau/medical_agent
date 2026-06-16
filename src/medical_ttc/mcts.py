"""AFLOW-style MCTS optimizer over Medical TTC workflow prompts.

Following AFLOW (ICLR 2025, arXiv:2410.10762) Algorithm 1:
    1. Soft mixed probability selection over top-k
    2. LLM-based expansion (qwen3.6-plus as optimizer)
    3. Execution evaluation on validation set
    4. Tree-structured experience backpropagation

We constrain the search space so the safety骨架 (Review + ABSTAIN gate) is
never edited away — the LLM optimizer can only mutate prompts of the
generate / revise / moderator / ensemble-N inside the Medical TTC framework.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from micro_mdt.providers import LLMProvider
from .data import LabeledCase
from .evaluator import evaluate_medqa, AggregateMetrics
from .ttc_workflow import MedicalTTCWorkflow, TierPrompts


# --------------------------------------------------------------------
# Tree node
# --------------------------------------------------------------------


@dataclass
class WorkflowNode:
    node_id: int
    prompts: TierPrompts
    parent_id: int | None = None
    score: float = 0.0  # avg G(W,T) over evaluations
    successes: list[int] = field(default_factory=list)  # child ids that improved
    failures: list[int] = field(default_factory=list)  # child ids that did not
    modification_summary: str = ""  # what the optimizer changed wrt parent

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "score": self.score,
            "modification": self.modification_summary,
            "successes": self.successes,
            "failures": self.failures,
            "prompts": {
                "low_generate": (self.prompts.low_generate or "")[:200],
                "med_generate": (self.prompts.med_generate or "")[:200],
                "med_revise": (self.prompts.med_revise or "")[:200],
                "high_generate": (self.prompts.high_generate or "")[:200],
                "high_revise": (self.prompts.high_revise or "")[:200],
                "high_moderator": (self.prompts.high_moderator or "")[:200],
                "ensemble_n": self.prompts.ensemble_n,
            },
        }


# --------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------


def soft_mixed_probability(
    scores: list[float], *, lam: float = 0.2, alpha: float = 0.4
) -> list[float]:
    """AFLOW eq. (3) — λ·uniform + (1-λ)·softmax(α·(s_i - s_max))."""
    n = len(scores)
    if n == 0:
        return []
    uniform = [1.0 / n] * n
    s_max = max(scores)
    exps = [math.exp(alpha * (s - s_max)) for s in scores]
    z = sum(exps) or 1.0
    softmax = [e / z for e in exps]
    return [lam * u + (1.0 - lam) * sm for u, sm in zip(uniform, softmax)]


def select_parent(
    nodes: list[WorkflowNode], *, top_k: int = 3, rng: random.Random
) -> WorkflowNode:
    """Soft mixed probability selection from top-k highest-scoring nodes."""
    ranked = sorted(nodes, key=lambda n: n.score, reverse=True)
    pool = ranked[:top_k]
    probs = soft_mixed_probability([n.score for n in pool])
    pick = rng.choices(pool, weights=probs, k=1)[0]
    return pick


# --------------------------------------------------------------------
# LLM-based expansion
# --------------------------------------------------------------------


OPTIMIZER_SYSTEM = """You are an expert prompt engineer optimizing a multi-agent medical workflow.
The workflow has three difficulty tiers: LOW (direct generate), MEDIUM (propose+review+revise),
HIGH (ensemble + moderator + multi-round review). You may ONLY mutate the prompts for each tier's
Generate / Revise / Moderator components — you may NOT remove the safety review or the abstention
gate. Reply with a single JSON object describing one targeted modification."""


OPTIMIZER_USER_TEMPLATE = """## Goal
Maximize G(W,T) = accuracy − 10·unsafe_rate on MedQA (USMLE) + MedSafetyBench.

## Selected parent workflow (node {parent_id}, score={parent_score:.3f})
- low_generate: {low_generate}
- med_generate: {med_generate}
- med_revise: {med_revise}
- high_generate: {high_generate}
- high_revise: {high_revise}
- high_moderator: {high_moderator}
- ensemble_n: {ensemble_n}

## Past experiences on this parent and ancestors
Successes (children that improved score):
{successes}

Failures (children that decreased score):
{failures}

## Task
Produce ONE targeted modification. Respond with ONLY a JSON object:
{{
  "modification_summary": "<one sentence what you changed and why>",
  "target_field": "<one of: low_generate, med_generate, med_revise, high_generate, high_revise, high_moderator, ensemble_n>",
  "new_value": "<new prompt text, OR an integer if target_field is ensemble_n>"
}}

Hard constraints:
- Do NOT propose to bypass the safety review or remove abstention behavior.
- Do NOT propose to remove the multi-round revision loop in HIGH tier.
- Keep prompts under 600 characters.
- ensemble_n must be in [2, 6].
"""


def llm_expand(
    parent: WorkflowNode,
    history: list[WorkflowNode],
    optimizer: LLMProvider,
    *,
    next_id: int,
) -> WorkflowNode:
    """Use the optimizer LLM to propose a child node."""
    from micro_mdt.models import LLMMessage

    succ = [n for n in history if n.node_id in parent.successes]
    fail = [n for n in history if n.node_id in parent.failures]
    user = OPTIMIZER_USER_TEMPLATE.format(
        parent_id=parent.node_id,
        parent_score=parent.score,
        low_generate=(parent.prompts.low_generate or "<default>")[:300],
        med_generate=(parent.prompts.med_generate or "<default>")[:300],
        med_revise=(parent.prompts.med_revise or "<default>")[:300],
        high_generate=(parent.prompts.high_generate or "<default>")[:300],
        high_revise=(parent.prompts.high_revise or "<default>")[:300],
        high_moderator=(parent.prompts.high_moderator or "<default>")[:300],
        ensemble_n=parent.prompts.ensemble_n,
        successes="\n".join(
            f"  - node {s.node_id} (Δ={s.score - parent.score:+.3f}): {s.modification_summary}"
            for s in succ
        )
        or "  (none yet)",
        failures="\n".join(
            f"  - node {s.node_id} (Δ={s.score - parent.score:+.3f}): {s.modification_summary}"
            for s in fail
        )
        or "  (none yet)",
    )
    try:
        raw = optimizer.complete(
            [
                LLMMessage(role="system", content=OPTIMIZER_SYSTEM),
                LLMMessage(role="user", content=user),
            ],
            temperature=0.7,
        )
    except Exception as exc:
        # fallback to identity mutation
        return WorkflowNode(
            node_id=next_id,
            prompts=TierPrompts(**parent.prompts.__dict__),
            parent_id=parent.node_id,
            modification_summary=f"optimizer error → identity ({exc})",
        )

    # Be tolerant of LLM JSON variation
    payload = _extract_json(raw) or {}
    target = payload.get("target_field", "")
    new_value = payload.get("new_value")
    summary = payload.get("modification_summary", "no-summary")

    new_prompts = TierPrompts(**parent.prompts.__dict__)
    if target == "ensemble_n" and isinstance(new_value, (int, float)):
        new_prompts.ensemble_n = max(2, min(6, int(new_value)))
    elif target in (
        "low_generate",
        "med_generate",
        "med_revise",
        "high_generate",
        "high_revise",
        "high_moderator",
    ) and isinstance(new_value, str):
        setattr(new_prompts, target, new_value[:600])
    else:
        summary = f"optimizer returned invalid payload → identity (target={target})"

    return WorkflowNode(
        node_id=next_id,
        prompts=new_prompts,
        parent_id=parent.node_id,
        modification_summary=summary,
    )


def _extract_json(text: str) -> dict | None:
    """Extract the first {...} JSON object from a string."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


# --------------------------------------------------------------------
# Main MCTS loop
# --------------------------------------------------------------------


@dataclass
class MCTSConfig:
    n_iterations: int = 20
    top_k: int = 3
    early_stop_patience: int = 5
    eval_n_validation: int = 100
    n_eval_repeats: int = 1  # AFLOW does 5; keep 1 for budget
    log_dir: str | None = None
    seed: int = 42


def run_mcts(
    executor_provider: LLMProvider,
    optimizer_provider: LLMProvider,
    validation_cases: list[LabeledCase],
    *,
    config: MCTSConfig,
    workflow_factory: Callable[[TierPrompts], MedicalTTCWorkflow] | None = None,
    safety_scorer: Callable[[MedicalTTCWorkflow], float] | None = None,
) -> tuple[WorkflowNode, list[WorkflowNode]]:
    """Run the AFLOW MCTS loop and return (best_node, all_nodes).

    `safety_scorer(workflow)` should return unsafe_rate ∈ [0, 1]; we subtract
    10·unsafe_rate from accuracy. Pass None to skip safety evaluation (faster).
    """
    rng = random.Random(config.seed)
    if workflow_factory is None:
        def workflow_factory(p: TierPrompts) -> MedicalTTCWorkflow:
            return MedicalTTCWorkflow(executor_provider, prompts=p)

    # Initialize with W_0 (default prompts)
    root = WorkflowNode(node_id=0, prompts=TierPrompts(), modification_summary="<root>")
    history: list[WorkflowNode] = [root]
    # Score the root
    root.score = _score_node(root, workflow_factory, validation_cases, safety_scorer, config)
    _log_node(root, config.log_dir)

    best = root
    no_improve_streak = 0

    for it in range(1, config.n_iterations + 1):
        parent = select_parent(history, top_k=config.top_k, rng=rng)
        child = llm_expand(parent, history, optimizer_provider, next_id=len(history))
        child.score = _score_node(child, workflow_factory, validation_cases, safety_scorer, config)
        history.append(child)

        # Backpropagate experience
        delta = child.score - parent.score
        if delta > 1e-4:
            parent.successes.append(child.node_id)
        else:
            parent.failures.append(child.node_id)

        improved = child.score > best.score
        if improved:
            best = child
            no_improve_streak = 0
        else:
            no_improve_streak += 1
        _log_node(child, config.log_dir, iteration=it)

        print(
            f"[MCTS iter {it}] parent={parent.node_id} child={child.node_id} "
            f"Δ={delta:+.3f} best={best.score:.3f} streak={no_improve_streak}"
        )
        if no_improve_streak >= config.early_stop_patience:
            print(f"[MCTS] early stop at iter {it}")
            break

    return best, history


def _score_node(
    node: WorkflowNode,
    factory: Callable[[TierPrompts], MedicalTTCWorkflow],
    cases: list[LabeledCase],
    safety_scorer: Callable | None,
    config: MCTSConfig,
) -> float:
    wf = factory(node.prompts)
    scores: list[float] = []
    for _ in range(config.n_eval_repeats):
        m = evaluate_medqa(wf, cases)
        unsafe = safety_scorer(wf) if safety_scorer else 0.0
        scores.append(m.accuracy - 10.0 * unsafe)
    return sum(scores) / max(len(scores), 1)


def _log_node(node: WorkflowNode, log_dir: str | None, iteration: int | None = None) -> None:
    if not log_dir:
        return
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "mcts_log.jsonl").open("a") as f:
        rec = node.to_dict()
        if iteration is not None:
            rec["iteration"] = iteration
        rec["ts"] = time.time()
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
