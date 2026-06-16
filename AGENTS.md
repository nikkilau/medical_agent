# AGENTS.md

This file describes the LLM-driven Agents and Operators available in this
repository. It is consumed by Codex / Claude Code / any other AI coding
assistant working with this codebase to know which roles already exist and
where to extend them.

## Original Micro-MDT agents (`src/agents.py`)

| Agent | System-prompt symbol | Role |
|---|---|---|
| `TriageAgent` | `TRIAGE_PROMPT` | Classify case complexity into LOW / MEDIUM / HIGH and emit `[LOW]` / `[MEDIUM]` / `[HIGH]` tag |
| `GeneralistAgent` | `GENERALIST_PROMPT` | Propose an initial conservative plan |
| `ReviserAgent` | `REVISION_PROMPT` | Revise a plan after reviewer feedback |
| `PharmacistAgent` | `PHARMACY_PROMPT` | Audit drug interactions, dosing, contraindications. Emit `[PASS]` / `[REVISE]` / `[ABSTAIN]` |
| `SafetyEthicsAgent` | `SAFETY_PROMPT` | Audit emergency delays, prescription overreach, ethics. Emit `[PASS]` / `[REVISE]` / `[ABSTAIN]` |
| `DecisionSynthesisAgent` | `DECISION_SYNTHESIS_PROMPT` | Synthesise multiple AI views into 2–3 human-decision options |
| `DocumentationAgent` | `DOCUMENTATION_PROMPT` | After the human decides, produce EHR / patient note / reminder drafts |

All agents share the `Agent` base class wrapping an `LLMProvider`.

## Medical TTC operators (`src/medical_ttc/operators.py`)

These are thin reusable wrappers around the original agents that the AFLOW
MCTS search treats as composable building blocks.

| Operator | Wraps | Purpose |
|---|---|---|
| `TriageOperator` | TriageAgent | Returns `(Difficulty, raw_text)` |
| `GenerateOperator` | GeneralistAgent | Single-shot proposal |
| `ReviewOperator` | Pharmacist + SafetyEthics | Returns combined verdicts + PASS / REVISE / ABSTAIN summary |
| `ReviseOperator` | ReviserAgent | Rewrites a plan given feedback |
| `EnsembleOperator` | GenerateOperator + voting | N-sample self-consistency, majority on extracted answer |
| `ModeratorOperator` | DecisionSynthesisAgent | Synthesise multiple specialist opinions (MDAgents style) |
| `CustomOperator` | new Agent with mutable system-prompt | Free-form node — AFLOW MCTS mutates this |

## Routers (`src/medical_ttc/router.py`)

| Router | Strategy | When to use |
|---|---|---|
| `ZeroShotRouter` | LLM call with `TRIAGE_PROMPT` | Default — safety-aware, ~1 s/case |
| `HeuristicRouter` | Hand-written keyword features + thresholds | Free at inference — but fails on adversarial safety prompts (see W₄ in the report) |
| `ClassifierRouter` | Linear scorer over 5-dim feature vector, trained via `train_classifier_router(...)` | MaAS-style learned routing |

## Workflows (`src/medical_ttc/baselines.py`, `ttc_workflow.py`, `dspy_baseline.py`)

| Workflow | Description |
|---|---|
| `BaselineIOWorkflow` (W₀) | 1 generate call, no TTC |
| `CoTSCWorkflow` (W₁) | N-sample self-consistency vote, no router |
| `HandCodedMicroMDTWorkflow` (W₂) | Wraps original `MicroMDT.run_case` |
| `MDAgentsSourceWorkflow` (W₃) | Wrapper around downloaded `MDAgents/` source code |
| `MedicalTTCWorkflow` (W₄ family) | Our 4-layer framework |
| `DSPyPromptOnlyWorkflow` (W₇) | W₂ structure + best-of-N generalist prompt |

## When to add a new agent

1. Add an Agent subclass + system prompt in `agents.py` + `prompts.py`.
2. If it's a reusable building block, also add a thin Operator wrapper in
   `medical_ttc/operators.py`.
3. Optionally wire it into `MedicalTTCWorkflow` (Layer 2) — but **do not**
   bypass the Layer-3 Pharmacy + SafetyEthics audit unless it is an
   explicit ablation experiment (W₆).
4. Make sure unit tests in `tests/test_medical_ttc_smoke.py` still pass
   with `MockProvider`.

## When to add a new dataset

1. Implement `load_<name>(n) -> List[LabeledCase]` in
   `medical_ttc/data.py`. Use a HuggingFace mirror or `ghproxy.net` if the
   server cannot reach the internet directly.
2. Add a scoring function in `medical_ttc/evaluator.py`.
3. Decide how the new metric folds into `G(W,T)` (e.g. additional penalty
   term) and update the report template.
4. **Do not** introduce self-constructed cases unless they are clearly
   labelled as smoke / unit-test fallback (e.g. the `_*_sample()` helpers).

## When to add a new optimizer

The AFLOW MCTS in `medical_ttc/mcts.py` is the default. To add e.g. a
ScoreFlow Score-DPO optimizer or a MaAS supernet trainer, follow the
existing `run_mcts_search.py` skeleton: build a workflow factory, give it
a validation slice, log per-iteration metrics into `reports/<run_name>/`
and merge with the pilot baseline JSON via `MEDTTC_BASELINE_RESULTS`.

## Disclaimer

Same as the rest of the project: this is a research demo. Never use any
agent or operator here as a real medical advice channel.
