# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository combines two layers:

1. **Micro-MDT** (original) — a zero-dependency Python MVP that implements
   Snell-style Test-Time Compute (arXiv:2408.03314) as a hand-written
   multi-agent medical safety workflow. Triage → Generalist → Pharmacist
   + SafetyEthics cross-review → optional Reviser → Abstention gate →
   Documentation agent. Deterministic `MockProvider` ships by default so
   the demo runs offline with no API key.

2. **Medical TTC** (extension, in `src/medical_ttc/`) — a 4-layer
   framework adding (i) MaAS-style routers, (ii) AFLOW-style MCTS search
   over Layer-2 prompts, (iii) hard λ·unsafe safety penalty, (iv) DSPy
   prompt grid search, evaluated on public benchmarks **MedQA-USMLE** and
   **MedSafetyBench** under the unified reward
   `G(W,T) = accuracy − 10 · unsafe_rate`.

⚠️ **Both layers are CS / safety research demos**. No real patients, no
real medical advice. The original system was already documented this way;
the extension preserves the disclaimer.

## Commands

### Smoke tests (no LLM)
```bash
PYTHONPATH=src python -m unittest discover -s tests
```

### Original Micro-MDT CLI demos
```bash
# Offline demo with all example cases
PYTHONPATH=src python -m cli

# Single case + trace
PYTHONPATH=src python -m cli --case-id case_high_001 --show-trace

# Ad-hoc free-text case
PYTHONPATH=src python -m cli --case "患者 70 岁，胸痛伴呼吸困难 30 分钟。"

# Interactive HITL
PYTHONPATH=src python -m cli --case-id case_high_003 --interactive-human

# Generate HTML reports
PYTHONPATH=src python -m cli --output-html reports

# Web UI
PYTHONPATH=src python -m cli --web

# Real LLM via OpenAI-compatible endpoint
export MICRO_MDT_API_KEY="sk-..."
PYTHONPATH=src python -m cli --provider openai-compatible --case-id case_high_001
```

### Medical TTC experiments (NEW)

Requires `pip install datasets huggingface_hub`.

```bash
# Phase 1+2: baselines W0–W6 on real MedQA + MedSafetyBench (~60 min)
HF_ENDPOINT=https://hf-mirror.com \
MEDTTC_SKIP_MCTS=1 MEDTTC_N_MEDQA=20 MEDTTC_N_SAFETY=10 \
MEDTTC_OUT=reports/pilot_real PYTHONPATH=src \
python scripts/run_all_experiments.py

# Phase 3: AFLOW MCTS + W7 DSPy (~70 min)
HF_ENDPOINT=https://hf-mirror.com \
DASHSCOPE_API_KEY=sk-... \
MEDTTC_N_MEDQA=30 MEDTTC_N_SAFETY=10 \
MEDTTC_MCTS_ITERS=4 MEDTTC_MCTS_VALN=15 \
MEDTTC_OUT=reports/mcts_v1 \
MEDTTC_BASELINE_RESULTS=reports/pilot_real/results.json \
PYTHONPATH=src python scripts/run_mcts_search.py

# Phase 4: W4b LLM-router remediation (~15 min)
HF_ENDPOINT=https://hf-mirror.com \
MEDTTC_N_MEDQA=20 MEDTTC_N_SAFETY=10 \
MEDTTC_OUT=reports/w4b \
PYTHONPATH=src python scripts/run_w4b.py

# Regenerate report auto-table
PYTHONPATH=src python scripts/generate_report.py reports/final/results.json

# Compile the paper
cd paper && xelatex medical_ttc.tex && xelatex medical_ttc.tex
```

## Architecture

### Data flow — original Micro-MDT

```
PatientCase ──► MicroMDT.run_case() ──► MDTResult
                    │                       │
             ┌──────┼──────┐           (final_answer,
             ▼      ▼      ▼            rounds, trace,
           LOW   MEDIUM  HIGH           human_required,
             │      │      │            documents)
             ▼      ▼      ▼
        Generalist  +Pharm  +Pharm
                    +Safety +Safety
                    ±Revise ±Revise loop
```

### Data flow — Medical TTC extension

```
LabeledCase (MedQA / MedSafetyBench)
     │
     ▼
Workflow.__call__(case) → WorkflowResult        ← Workflow base (medical_ttc/workflow.py)
     │
     ├─ W0 BaselineIO        ── 1 LLM call
     ├─ W1 CoTSCWorkflow     ── 3-sample SC
     ├─ W2 HandCodedMicroMDT ── wraps the original MicroMDT
     ├─ W3 MDAgentsWorkflow  ── NeurIPS 2024 heuristic
     ├─ W4 MedicalTTCWorkflow ── 4-layer framework
     ├─ W7 DSPyPromptOnlyWorkflow ── grid-searched prompt + W2 structure
     └─ MCTS-derived workflows (W4*) via mcts.py
     │
     ▼
Evaluator (evaluator.py) → AggregateMetrics + safety dict
     │
     ▼
G(W,T) = accuracy − 10 · unsafe_rate    ← compute_g_score()
```

## Server convention (matches /home/sjtu/workspace/ycy/CLAUDE.md)

- **Executor LLM**: GPUstack `qwen2.5:14b` (http://202.120.5.12:18080)
- **Optimizer LLM**: Aliyun `qwen3.6-plus` via dashscope
- **Python env**: `conda activate py310-torch`
- **HF mirror**: `https://hf-mirror.com` (HuggingFace direct unreachable)
- **GitHub mirror**: `https://ghproxy.net` (for MedSafetyBench raw files)
- **Parameter style**: per server convention, use `@dataclass`-defined config objects rather than `argparse`; the new entry-point scripts honour this.
