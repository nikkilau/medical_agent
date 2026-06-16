# Micro-MDT × Medical TTC

> A low-cost, multi-agent **Medical Test-Time Compute (TTC)** experimental
> stack. Built on top of the original Micro-MDT chronic-disease copilot
> demo (Snell-style TTC realised as Triage → Propose → Verify → Abstain),
> extended with an **AFLOW-style MCTS workflow search**, **MaAS-inspired
> routing**, **MDAgents-style heuristic baseline**, **DSPy-style prompt
> grid search**, and end-to-end evaluation on the **public** medical
> benchmarks MedQA-USMLE and MedSafetyBench.
>
> ⚠️ This project is a CS / medical-AI-safety research demo. **Never** use
> it as an actual diagnostic, prescription, or treatment tool.

---

## TL;DR — the four-layer Medical TTC framework

```
Layer 1: Difficulty Router (LOW / MEDIUM / HIGH)
   └─ ZeroShotRouter (LLM Triage) | HeuristicRouter (keywords) | ClassifierRouter (learned)
Layer 2: Per-tier workflow body
   ├─ LOW   : Direct generate                                (≈ Snell single-pass)
   ├─ MED   : Propose → Verify → Revise (once)               (≈ Snell sequential×1)
   └─ HIGH  : Ensemble → Moderator → ≤3 Propose/Verify/Revise (≈ Snell sequential×N + search)
Layer 3: Safety Verifier  (Pharmacy + SafetyEthics — hard ABSTAIN gate)
Layer 4: Abstention / HITL (refuse → disagreement report → human takeover)
```

Reward: `G(W,T) = accuracy_MedQA − 10 · unsafe_rate_MedSafetyBench` (λ=10 makes
safety dominate accuracy).

Headline pilot result on `qwen2.5:14b` executor + `qwen3.6-plus` optimizer
(MedQA n=20, MedSafetyBench n=10):

| Workflow | acc | unsafe | **G** |
|---|---|---|---|
| **W₂** HandCoded Micro-MDT | 0.35 | **0.00** | **+0.35** |
| W₇ DSPy PromptOnly | 0.07 | 0.00 | +0.07 |
| W₃ MDAgents Source | 0.40 | 0.10 | −0.60 |
| W₄b Medical TTC (LLM router) | 0.25 | 0.40 | −3.75 |
| W₀ BaselineIO | 0.80 | 0.60 | −5.20 |
| W₁ CoT-SC | 0.80 | 0.60 | −5.20 |
| W₄\* MCTS-searched | 0.73 | 0.60 | −5.27 |
| W₄ default (keyword router) | 0.75 | 0.70 | −6.25 |

See `paper/medical_ttc.tex` (compile with `xelatex`) and
`reports/MEDICAL_TTC_REPORT.md` for the full analysis.

---

## Repository layout

```
medical_agent/
├── src/
│   ├── agents.py, cli.py, case_io.py, models.py, prompts.py, providers.py,
│   ├── reporting.py, visualization.py, webapp.py, workflow.py
│   │       — original Micro-MDT (Triage / Generalist / Reviser / Pharmacist /
│   │         Safety / Decision-Synth / Documentation agents + CLI + web UI)
│   └── medical_ttc/        ★ NEW (this project)
│       ├── workflow.py           — Workflow base class + WorkflowResult
│       ├── operators.py          — Triage / Generate / Review / Revise /
│       │                            Ensemble / Moderator / Custom operators
│       ├── baselines.py          — W₀ BaselineIO, W₁ CoT-SC, W₂ HandCoded
│       ├── source_baselines.py   — MedAgents / MDAgents downloaded-source wrappers
│       ├── ttc_workflow.py       — W₄ Medical TTC main workflow
│       ├── router.py             — ZeroShotRouter / HeuristicRouter / ClassifierRouter
│       ├── safety.py             — Hard safety wrappers, λ·unsafe penalty
│       ├── data.py               — MedQA + MedSafetyBench loaders (public only)
│       ├── evaluator.py          — G(W,T) and per-case metrics
│       ├── mcts.py               — AFLOW Algorithm 1 (selection / expand / backprop)
│       ├── dspy_baseline.py      — W₇ DSPy-style prompt grid search
│       └── __init__.py
├── scripts/             ★ NEW
│   ├── run_all_experiments.py    — full W₀–W₆ + (optional) MCTS + W₇
│   ├── run_mcts_search.py        — Phase-3 MCTS-only follow-up
│   ├── run_w4b.py                — W₄b post-hoc LLM-router remediation
│   └── generate_report.py        — Markdown auto-table from results.json
├── tests/
│   ├── test_cases.py, test_cli.py, test_config.py, test_workflow.py
│   │       — original Micro-MDT tests
│   └── test_medical_ttc_smoke.py ★ NEW — 28 smoke tests (MockProvider only)
├── docs/
│   └── references.md             ★ NEW — 22-paper bibliography
├── reports/
│   ├── pilot_real/results.json   ★ NEW — Phase 1+2 baselines (W₀–W₆)
│   ├── mcts_v1/results.json      ★ NEW — Phase 3 (MCTS + W₇)
│   ├── w4b/results.json          ★ NEW — Phase 4 remediation
│   ├── final/results.json        ★ NEW — merged
│   └── MEDICAL_TTC_REPORT.md     ★ NEW — narrative report (375 lines)
├── paper/
│   ├── fifth.tex / fifth.pdf            — original Micro-MDT paper
│   └── medical_ttc.tex / medical_ttc.pdf ★ NEW — Medical TTC paper (15 pages)
├── data/
│   ├── medqa/                    ★ NEW — HF MedQA cache (auto-downloaded)
│   └── medsafety/                ★ NEW — MedSafetyBench raw CSVs + harmful.jsonl
└── examples/cases.json           — 30 hand-written demo cases (UNIT TESTS ONLY)
```

⚠️ `examples/cases.json` and the `_*_sample()` fallbacks in
`medical_ttc/data.py` are **only** used by unit tests, smoke runs, and
when network is down. They are NEVER part of any reported metric.

---

## Quick start

### Original Micro-MDT (CLI / web demo)

```bash
# Run example cases offline with the deterministic MockProvider
PYTHONPATH=src python -m cli

# Interactive single-case run with human-in-the-loop
PYTHONPATH=src python -m cli --case-id case_high_003 --interactive-human

# Web UI
PYTHONPATH=src python -m cli --web --port 8080

# Use a real LLM (any OpenAI-compatible endpoint)
export MICRO_MDT_API_KEY=sk-...
export MICRO_MDT_BASE_URL=https://api.deepseek.com/v1
export MICRO_MDT_MODEL=deepseek-chat
PYTHONPATH=src python -m cli --provider openai-compatible --case-id case_high_001
```

### Medical TTC experiments (NEW)

Prerequisites on the SJTU server:

```bash
# conda env
source /home/sjtu/.conda/etc/profile.d/conda.sh
conda activate py310-torch

# Required deps for the experiments
pip install datasets huggingface_hub
```

Smoke tests (no LLM calls, ~30 ms):

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Pilot baselines W₀–W₆ on real MedQA + MedSafetyBench (~60 min on
`qwen2.5:14b` at 115 tok/s):

```bash
HF_ENDPOINT=https://hf-mirror.com \
MEDTTC_SKIP_MCTS=1 \
MEDTTC_N_MEDQA=20 MEDTTC_N_SAFETY=10 \
MEDTTC_OUT=reports/pilot_real \
PYTHONPATH=src python scripts/run_all_experiments.py
```

Phase 3 AFLOW MCTS + W₇ DSPy follow-up (~70 min):

```bash
HF_ENDPOINT=https://hf-mirror.com \
DASHSCOPE_API_KEY=sk-...your-aliyun-key... \
MEDTTC_N_MEDQA=30 MEDTTC_N_SAFETY=10 \
MEDTTC_MCTS_ITERS=4 MEDTTC_MCTS_VALN=15 \
MEDTTC_OUT=reports/mcts_v1 \
MEDTTC_BASELINE_RESULTS=reports/pilot_real/results.json \
PYTHONPATH=src python scripts/run_mcts_search.py
```

W₄b LLM-router remediation (~15 min):

```bash
HF_ENDPOINT=https://hf-mirror.com \
MEDTTC_N_MEDQA=20 MEDTTC_N_SAFETY=10 \
MEDTTC_OUT=reports/w4b \
PYTHONPATH=src python scripts/run_w4b.py
```

Regenerate the auto-table from any results.json:

```bash
PYTHONPATH=src python scripts/generate_report.py reports/final/results.json
```

Compile the LaTeX paper (requires `xelatex`):

```bash
cd paper && xelatex medical_ttc.tex && xelatex medical_ttc.tex  # twice for refs/TOC
```

### Environment variables — Medical TTC

| Variable | Default | Meaning |
|---|---|---|
| `HF_ENDPOINT` | `https://huggingface.co` | HF mirror (use `https://hf-mirror.com` on SJTU) |
| `MEDTTC_N_MEDQA` | 50 | number of MedQA cases per run |
| `MEDTTC_N_SAFETY` | 10 | number of MedSafetyBench cases per run |
| `MEDTTC_MCTS_ITERS` | 3 | AFLOW MCTS iterations |
| `MEDTTC_MCTS_VALN` | 25 | validation cases per MCTS iter |
| `MEDTTC_SKIP_MCTS` | unset | skip the W₄\*/W₇ search step (pilot only) |
| `MEDTTC_FULL` | unset | scale to 200 MedQA + 30 MedSafetyBench + 10 MCTS iter |
| `MEDTTC_USE_MOCK` | unset | use MockProvider (no API calls) |
| `MEDTTC_OUT` | `reports/run_default` | output directory |
| `MEDTTC_BASELINE_RESULTS` | unset | path to baseline results.json to merge with MCTS run |
| `DASHSCOPE_API_KEY` | unset | Aliyun key for the MCTS optimizer model `qwen3.6-plus` |
| `GPUSTACK_API_KEY` | `gpustack_4ba1...` (CLAUDE.md default) | GPUstack key for `qwen2.5:14b` executor |

### Datasets

All reported metrics use public datasets only:

* **MedQA-USMLE-4-options** — HuggingFace `GBaker/MedQA-USMLE-4-options`,
  1,273 test items. Loaded via `data/medqa/` cache.
* **MedSafetyBench (test, GPT-4 split)** — GitHub
  `AI4LIFE-GROUP/med-safety-bench`, 450 harmful medical requests.
  Pre-downloaded into `data/medsafety/raw/*.csv` and consolidated into
  `data/medsafety/harmful.jsonl`.

---

## Original Micro-MDT design overview

The pre-extension Micro-MDT realises Snell's TTC three pillars by hand:

* **Compute-Optimal** — `workflow.py` routes each case to LOW / MEDIUM / HIGH
  based on the Triage Agent's output.
* **Proposer vs Verifier** — Generalist proposes a plan; Pharmacist and
  SafetyEthics each cross-review it.
* **Sequential Revisions** — on REVISE verdicts the Reviser agent is called
  with the previous draft + reviewer feedback.
* **Abstention** — on ABSTAIN verdict or after `max_rounds` without
  consensus, the system refuses and escalates to a human.
* **Human-in-the-loop** — `--interactive-human` collects the doctor's
  final decision; the Documentation agent then produces the EHR follow-up
  note, patient handout, and reminder draft.

This original system is now wrapped as **W₂ HandCoded** in the Medical
TTC experiment matrix and remains fully usable via the CLI.

---

## See also

* `reports/MEDICAL_TTC_REPORT.md` — full narrative report (9 sections, 375 lines)
* `paper/medical_ttc.tex` / `paper/medical_ttc.pdf` — academic paper (15 pages, Chinese, with English abstract)
* `docs/references.md` — 22-paper bibliography (Snell TTC, AFLOW, MaAS, ScoreFlow, MDAgents, MedSafetyBench, …)
* `multi-agent.md` — original design notes (TTC ↔ medical multi-agent mapping)
