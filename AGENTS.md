# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

Micro-MDT is a zero-dependency Python MVP that implements Test-Time Compute theory (arXiv:2408.03314) as a multi-agent medical safety workflow. It simulates a Multi-Disciplinary Team (MDT) with Proposer/Verifier debate, sequential revision, and safety abstention — but uses **no real patients, no training, and no real medical advice**. The default `MockProvider` is deterministic and requires no API key.

## Commands

```powershell
# Run all example cases (mock provider, no API key needed)
$env:PYTHONPATH="src"
python -m micro_mdt.cli

# Run a specific case from examples/cases.json
$env:PYTHONPATH="src"
python -m micro_mdt.cli --case-id case_high_001 --show-trace

# Run an ad-hoc case from text
$env:PYTHONPATH="src"
python -m micro_mdt.cli --case "患者 70 岁，胸痛伴呼吸困难 30 分钟。"

# Interactive human-in-the-loop mode
$env:PYTHONPATH="src"
python -m micro_mdt.cli --case-id case_high_003 --interactive-human

# Generate HTML reports
$env:PYTHONPATH="src"
python -m micro_mdt.cli --output-html reports

# Start web UI
$env:PYTHONPATH="src"
python -m micro_mdt.cli --web

# Use real LLM (OpenAI-compatible API)
$env:PYTHONPATH="src"
$env:MICRO_MDT_API_KEY="sk-..."
python -m micro_mdt.cli --provider openai-compatible --case-id case_high_001

# Run tests
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

The `$env:PYTHONPATH="src"` prefix is only needed when running outside an installed package. If installed via `pip install -e .`, run commands directly (e.g., `micro-mdt`).

## Architecture

### Data flow

```
PatientCase ──► MicroMDT.run_case() ──► MDTResult
                    │                       │
             ┌──────┼──────┐           (final_answer,
             ▼      ▼      ▼            rounds, trace,
           LOW   MEDIUM  HIGH           human_required,
             │      │      │            documents)
             ▼      ▼      ▼
        Generalist  +Pharm  +Pharm
        (direct)    +Safety +Safety
                    +1 rev  +≤max_rounds
                             debate
```

### Difficulty routing (compute-optimal)

- **LOW**: Triage → Generalist → output (1 LLM call beyond triage). No review.
- **MEDIUM**: Triage → Generalist → Pharmacist + Safety. If both pass → done (3 calls). If either fails → 1 revision + re-review (5 calls). If still failing → abstain.
- **HIGH**: Triage → Generalist → debate loop (Pharmacist + Safety → Revision) up to `max_rounds` (default 3). Consensus → output; persistent disagreement → abstain with structured A/B/C options.

### Key files

| File | Role |
|---|---|
| `src/micro_mdt/workflow.py` | Core orchestration: `MicroMDT.run_case()`, `_run_medium()`, `_run_high()`, abstention, human callback, document generation |
| `src/micro_mdt/agents.py` | `Agent` (prompt + provider → `run()`), `AgentSuite` (7 agents), `parse_difficulty()` / `parse_verdict()` |
| `src/micro_mdt/models.py` | Dataclasses: `PatientCase`, `AgentResponse`, `DebateRound`, `MDTResult`; enums: `Difficulty`, `Verdict` |
| `src/micro_mdt/prompts.py` | System prompts for all 7 agents + `DISCLAIMER` constant |
| `src/micro_mdt/providers.py` | `LLMProvider` (ABC), `MockProvider` (keyword-matching, deterministic), `OpenAICompatibleProvider` (stdlib `urllib`) |
| `src/micro_mdt/cli.py` | argparse CLI entry point, `ask_human_decision()` interactive callback |
| `src/micro_mdt/io.py` | Load cases from `examples/cases.json` or from ad-hoc text |
| `src/micro_mdt/reporting.py` | Terminal markdown-style output |
| `src/micro_mdt/visualization.py` | HTML report generation with flow diagrams, debate timelines, collapsible traces |
| `src/micro_mdt/webapp.py` | Interactive web UI on `http://localhost:8080` using stdlib `http.server`; sessions stored in `_SESSIONS` dict |

### The 7 agents (AgentSuite)

1. **Triage** — classifies case as `[LOW]`, `[MEDIUM]`, or `[HIGH]`
2. **Generalist** (Proposer) — generates initial treatment plan
3. **GeneralistRevision** — revises plan based on review feedback
4. **Pharmacist** (Verifier) — checks drug interactions, outputs `[PASS]`/`[REVISE]`/`[ABSTAIN]`
5. **SafetyEthics** (Verifier) — checks for emergency delay, harm risk, outputs `[PASS]`/`[REVISE]`/`[ABSTAIN]`
6. **DecisionSynthesis** — generates A/B/C structured options for human doctor when AI deadlocks
7. **Documentation** — generates EHR draft, patient note, and follow-up reminder from human decision

### Verdict parsing

`parse_verdict()` in `agents.py` matches `[PASS]`, `[ABSTAIN]` in agent output; default is `REVISE`. `parse_difficulty()` matches `[HIGH]`, `[MEDIUM]`; default is `LOW`. Both are case-insensitive and look for the bracket-delimited markers.

### MockProvider design

`MockProvider.complete()` dispatches to keyword-based methods by inspecting the system prompt for Chinese role names (e.g., "分诊 Agent" → `_triage()`). It checks for high-risk terms with negation awareness via `_has_any_unnegated()`. This is the default provider — no network or API key needed.

### Human-in-the-loop pattern

`MicroMDT.run_case()` accepts an optional `human_decision_callback: Callable[[MDTResult], str | None]`. When a case triggers abstention:
1. `DecisionSynthesis` agent generates A/B/C options → stored in `result.human_options`
2. If callback provided, it's called with the result; return value becomes `result.human_decision`
3. With a decision, `Documentation` agent generates three documents (EHR draft, patient note, reminder)
4. CLI provides `ask_human_decision()` as the callback; web app stores result in session and waits for a second POST

### Provider selection

CLI `--provider` flag: `mock` (default, zero-cost) or `openai-compatible`. The `OpenAICompatibleProvider` reads from env vars: `MICRO_MDT_API_KEY` (or `OPENAI_API_KEY`), `MICRO_MDT_BASE_URL`, `MICRO_MDT_MODEL`. It uses only stdlib `urllib` — no `requests`, no `openai` package.

### Testing

Tests use `unittest` with `MockProvider` (no network). All workflow paths are tested: LOW direct, MEDIUM revision, HIGH abstention, human decision → documentation, HTML rendering. Test file: `tests/test_workflow.py` — two classes, `MicroMDTWorkflowTests` and `VisualizationTests`.
