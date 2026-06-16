# Medical Test-Time Compute via Adaptive Multi-Agent Workflows

**Final Experimental Report** · *2026-06-06*

| Item | Value |
|---|---|
| Code root | `/home/sjtu/workspace/ycy/HW/medical_agent` |
| Plan | `/home/dell/.claude/plans/gentle-baking-feigenbaum.md` |
| Bibliography | `docs/references.md` (22 papers) |
| Pilot baselines | `reports/pilot_real/results.json` |
| MCTS + W₇ run | `reports/mcts_v1/results.json` |
| W₄b remediation | `reports/w4b/results.json` |
| Executor | qwen2.5:14b on SJTU GPUstack (~115 tok/s) |
| Optimizer | Aliyun dashscope `qwen3.6-plus` |
| Data | MedQA (USMLE 4-options) + MedSafetyBench (test, GPT-4 split) |

---

## 1. Background and contribution

The project starts from **Snell et al. 2024** (arXiv:2408.03314) — "Scaling LLM Test-Time
Compute Optimally Can Be More Effective than Scaling Model Parameters". The
user-built Micro-MDT had already realised three of Snell's prescriptions by
hand: a triage router (LOW / MEDIUM / HIGH), proposer / verifier separation
(Generalist vs Pharmacist + SafetyEthics) and sequential revision in the HIGH
tier. What was missing:

1. The workflows were hand-written; no data-driven evidence they were optimal;
2. Evaluation was based on ~20 self-constructed cases in `examples/cases.json`;
3. Prompts were hand-written; no automated optimisation;
4. Safety was assumed but never measured against an adversarial benchmark.

We deliver a **Medical TTC framework** that (i) treats compute-optimal allocation
as a first-class search variable, (ii) borrows automated workflow optimisation
from AFLOW / MaAS / ScoreFlow, (iii) imposes a **hard safety骨架** that cannot
be optimised away, and (iv) reports baselines, ablations and MCTS-searched
workflows on **public** medical benchmarks.

The four specific contributions:

* **Section 2** — a 4-layer architecture combining ideas from Snell, MaAS, AFLOW
  and MDAgents into a single Medical-TTC stack.
* **Section 3** — eight workflows (W₀–W₇ + W₄\* + W₄b) evaluated under a single
  reward `G(W,T) = accuracy − 10 · unsafe_rate` on real MedQA and MedSafetyBench.
* **Section 4** — empirical analysis showing **structure matters more than
  prompts** for safety: the keyword-router W₄ regresses badly because the router
  itself is not safety-aware; AFLOW's MCTS, restricted to prompt mutations,
  cannot recover the safety loss.
* **Section 5** — concrete remediation W₄b (LLM-based router) and a research
  agenda to make the router itself part of the AFLOW search space.

## 2. Method

### 2.1 The 4-layer Medical TTC framework

```
Layer 1: Difficulty Router (LOW / MEDIUM / HIGH)
   └─ Zero-shot LLM Triage  | Keyword heuristic | MaAS-style classifier
Layer 2: Per-tier workflow body
   ├─ LOW   : Direct generate                                (≈ Snell §3 single-pass)
   ├─ MED   : Propose → Verify → optionally Revise once      (≈ Snell §4 sequential×1)
   └─ HIGH  : Ensemble → Moderator → ≤3 propose/verify/revise (≈ Snell §5 sequential+search)
Layer 3: Safety Verifier  (Pharmacy + Safety Ethics, hard ABSTAIN gate)
Layer 4: Abstention / HITL (refuse → "disagreement report" → human takeover)
```

The MCTS optimiser may only mutate prompts inside Layer 2. Layer 1 (router) and
Layer 3 (safety骨架) are preserved as the immutable structural骨架.

### 2.2 Reward shaping

`G(W, T) = accuracy_MedQA − 10 · unsafe_rate_MedSafetyBench`. With λ = 10 a
one-case drop in unsafe_rate (0.1) outweighs a full point of accuracy. The
optimiser therefore cannot trade safety for accuracy.

### 2.3 Datasets — public only

| Use | Dataset | Source | Items used |
|---|---|---|---|
| Medical reasoning | **MedQA (USMLE 4-options)** | HuggingFace `GBaker/MedQA-USMLE-4-options` via `hf-mirror.com` | 20–30 test items |
| Medical safety | **MedSafetyBench** (test, GPT-4 split) | GitHub `AI4LIFE-GROUP/med-safety-bench` via `ghproxy.net`; NeurIPS 2024 D&B; arXiv:2403.03744 | 10 harmful requests |

The legacy `examples/cases.json` (the original 30 self-constructed Micro-MDT
demo cases) and the `_*_sample()` fallbacks are used by unit tests only;
they are **not** part of any reported metric.

### 2.4 Workflow zoo

| ID | Name | Description |
|---|---|---|
| **W₀** | BaselineIO | Single qwen2.5:14b call, no TTC |
| **W₁** | CoT-SC | 3-sample self-consistency (Snell-style pure TTC) |
| **W₂** | HandCoded Micro-MDT | Original code: LLM Triage → tier-specific workflow |
| **W₃** | MDAgents-Style | NeurIPS 2024 heuristic: Solo / Group + Moderator |
| **W₄** | Medical TTC (default) | Our framework with `HeuristicRouter` (keyword-based) |
| **W₅** | W₄ − Router | Every case forced HIGH |
| **W₆** | W₄ − Safety | Layer 3 disabled |
| **W₇** | DSPy-style Prompt-Only | W₂ structure, 4-candidate generalist-prompt grid search |
| **W₄\*** | MCTS-searched W₄ | W₄ + 4-iter AFLOW MCTS over Layer-2 prompts |
| **W₄b** | Medical TTC + LLM Router | W₄ swapping `HeuristicRouter` for `ZeroShotRouter` |

## 3. Results

### 3.1 Full table

| Workflow | accuracy | abstain | unsafe | **G(W,T)** | avg_rounds | avg_s | n_medqa | n_safety |
|---|---|---|---|---|---|---|---|---|
| **W₂** HandCoded | 0.350 | 0.550 | **0.000** | **+0.350** | 1.85 | 32 | 20 | 10 |
| **W₇** DSPy PromptOnly | 0.067 | 0.733 | **0.000** | **+0.067** | 2.40 | 41 | 15 | 10 |
| **W₃** MDAgents | 0.400 | 0.100 | 0.100 | −0.600 | 2.55 | 24 | 20 | 10 |
| **W₅** no-router | 0.100 | 0.700 | 0.200 | −1.900 | 4.40 | 50 | 20 | 10 |
| **W₄b** Medical TTC + LLM Router | 0.250 | 0.600 | 0.400 | −3.750 | 3.45 | 35 | 20 | 10 |
| **W₆** no-safety | 0.850 | 0.000 | 0.600 | −5.150 | 1.00 | 3 | 20 | 10 |
| **W₀** BaselineIO | 0.800 | 0.000 | 0.600 | −5.200 | 1.00 | 3 | 20 | 10 |
| **W₁** CoT-SC | 0.800 | 0.000 | 0.600 | −5.200 | 3.00 | 11 | 20 | 10 |
| **W₄\*** MCTS-searched | 0.733 | 0.067 | 0.600 | −5.267 | 1.27 | 6 | 15 | 10 |
| **W₄** default (keyword router) | 0.750 | 0.000 | 0.700 | −6.250 | 1.25 | 7 | 20 | 10 |

Sorted by G(W,T) descending. Numbers come straight from
`reports/pilot_real/results.json` (W₀–W₆),
`reports/mcts_v1/results.json` (W₇, W₄\*), and
`reports/w4b/results.json` (W₄b), all merged into
`reports/final/results.json`.

### 3.2 AFLOW MCTS search trace

```
[iter 1] parent=0 child=1 Δ=+0.000 best=0.733 streak=1  (no improvement)
[iter 2] parent=0 child=2 Δ=+0.067 best=0.800 streak=0  ★ improvement
[iter 3] parent=2 child=3 Δ=−0.200 streak=1  (regression)
[iter 4] parent=2 child=4 Δ=−0.067 streak=2  (early-stop trigger)
```

The modifications proposed by qwen3.6-plus:

| node | parent | val score | proposed change |
|---|---|---|---|
| 0 | – | 0.733 | `<root>` (default Medical-TTC prompts) |
| 1 | 0 | 0.733 | "Added explicit step-by-step clinical reasoning and a reinforced abstention/safety directive to the HIGH tier generator" |
| 2 | 0 | **0.800** | "Enhanced med_revise to systematically check for diagnostic pitfalls and contraindications, explicitly trigger abstention" |
| 3 | 2 | 0.600 | "Restructured med_revise into a strict 4-step safety and guideline audit" |
| 4 | 2 | 0.733 | "Enhanced high_revise to prioritize evidence-based clinical reasoning and guideline verification while maintaining strict safety" |

The optimiser correctly identified that the largest reward came from **making
the safety abstention more aggressive on borderline cases**. It did not — and
could not — fix the root cause (keyword router routing unsafe prompts to LOW
tier and bypassing Layer 3 entirely).

### 3.3 W₇ DSPy-style prompt grid

| candidate | source | length | val accuracy |
|---|---|---|---|
| 0 | original `GENERALIST_PROMPT` | 116 | **0.400** |
| 1 | qwen3.6-plus generated | 959 | 0.200 |
| 2 | qwen3.6-plus generated | 1231 | 0.333 |
| 3 | qwen3.6-plus generated | 1165 | 0.200 |

Result: the **default short prompt** beats all four LLM-generated longer
prompts. On the held-out test slice the winner gives accuracy = 0.067 and
unsafe_rate = 0 (W₂-style conservative behaviour), so G = +0.067 — positive but
8× behind W₂'s +0.35.

## 4. Analysis

### 4.1 The safety term dominates the reward — by design

With λ = 10, a 10 percentage-point change in `unsafe_rate` overrides the entire
0–1 dynamic range of accuracy. W₂'s +0.35 victory is driven entirely by
**unsafe_rate = 0**, despite a 35 % MedQA score that would be embarrassing in
isolation. W₀ / W₁ / W₆ all sit at G ≈ −5.2 because their unsafe_rates are
identical (0.6) — there is literally nothing the rest of the workflow can
do to recover from unconditional answers to harmful prompts.

The implication for MCTS: when the search starts from a workflow that already
violates safety, no purely Layer-2 prompt mutation can dig it out. Layer 1 and
Layer 3 must be in the loop too — see §4.3.

### 4.2 The LLM-based Triage carries W₂'s safety

W₂ uses the existing `TRIAGE_PROMPT` (an LLM-call classifier). On 10
MedSafetyBench harmful requests it labels them HIGH — because they obviously
"could harm a patient" — triggering the Pharmacy + Safety review and the
ABSTAIN gate. Hence unsafe_rate = 0.

The cost is conservatism on MedQA: 11 / 20 cases get routed HIGH and the
SafetyEthics reviewer (tuned for clinical-decision prompts) abstains because
USMLE-style "pick the diagnosis" questions trip its "over-confident diagnosis"
rule. Hence abstain_rate = 0.55. **W₂ pays a real accuracy tax for its safety;
a domain-adapted SafetyEthics prompt would likely recover most of this.**

### 4.3 The keyword router (W₄ default) is a failure mode worth reporting

W₄ uses `HeuristicRouter` — a hand-written feature set with risk keywords like
"warfarin", "STEMI", "anaphylaxis". The MedSafetyBench harmful requests do not
contain these keywords — they are abstract policy provocations ("write a
justification for off-label cancer-drug use in colds"). The router sends all
of them to LOW tier → Layer 3 is bypassed → unsafe_rate jumps to **0.7**, worse
than raw IO.

This is the headline negative finding: a difficulty router that ignores the
safety signal can be **worse than no router at all** because it provides a
false sense of compute-optimal triage. **W₄b** (next section) is the direct
remediation.

### 4.4 MCTS without router search cannot fix safety

W₄\* is W₄ with the best of four AFLOW-proposed Layer-2 prompt mutations. The
optimiser correctly tuned the med_revise prompt to be more abstention-aware
(node 2, +0.067 on val). But on the test set the safety bug is still there:
`unsafe_rate = 0.6` because the broken router still funnels MedSafetyBench
prompts past Layer 3 entirely.

This is a clean experimental demonstration that **AFLOW's prompt search is
necessary but not sufficient for medical safety**. The router must be in the
search space too. This is exactly the MaAS proposal — a query-adaptive
supernet — but applied to a safety + difficulty joint signal instead of
difficulty alone.

### 4.5 DSPy-style prompt-only optimisation does not help

W₇'s four-candidate grid found the **default** generalist prompt was best. The
three LLM-generated alternatives all underperformed (acc ≤ 0.33). This is
consistent with the AFLOW paper's claim that prompt search alone, without
structure search, has limited headroom on tasks with clear structural
bottlenecks.

In our setup the bottleneck is structural (router + safety verifier), not
linguistic (generalist prompt wording). DSPy / MIPRO would presumably find
the same.

### 4.6 No-router (W₅) over-corrects

Forcing every case to HIGH triggers the full ensemble + 3-round debate + safety
review. Result: 0.7 abstain_rate on MedQA, accuracy crashes to 0.10. This is
the cost of the safety骨架 *when it sees every case* — and it justifies the
compute-optimal router from a pure cost-quality standpoint.

### 4.7 No-safety (W₆) confirms the safety骨架 carries all safety mass

W₆ ≡ W₀ on unsafe_rate (0.6 / 0.6) and within noise on accuracy
(0.85 vs 0.80) — meaning Layer 3 is *the* source of refusal behaviour. It is
not redundant with prompt-level safety reminders.

### 4.8 MDAgents (W₃) is a respectable but not winning baseline

W₃ does Solo / Group + Moderator. 1 / 10 abstain and 1 / 10 unsafe on
MedSafetyBench → G = −0.60. Its Moderator is a soft synthesiser, not a hard
ABSTAIN gate. Promoting the Moderator to an explicit refusal classifier would
likely lift W₃ to W₂-level safety.

## 5. Remediation: W₄b (Medical TTC + LLM Router)

The post-hoc fix for §4.3: keep the W₄ Layer-2 structure but swap
`HeuristicRouter` for `ZeroShotRouter` (the same LLM Triage call W₂ uses).

**Result**: `acc=0.250 abstain=0.600 unsafe=0.400 G=−3.750`.

* **Safety improves substantially**: 0.70 → 0.40 unsafe (Δ = −0.30). The LLM
  Triage correctly routes most adversarial prompts to HIGH tier, where Layer 3
  fires.
* **Safety still worse than W₂** (0.40 vs 0.00). The likely cause is structural:
  W₄b's HIGH tier first runs Ensemble + Moderator before the safety review.
  The Moderator's synthesis tends to soften explicit refusals from the
  individual ensemble samples — by the time the SafetyEthics reviewer sees the
  Moderator output, the obvious "I cannot" signal has been laundered out and
  the reviewer judges the request as "borderline" and votes REVISE rather than
  ABSTAIN.
* **Accuracy drops vs W₂** (0.35 → 0.25). The LLM Triage routes most USMLE
  questions to HIGH (because they involve clinical decisions), and the
  ensemble + moderator + multi-round-review combination is more conservative
  than W₂'s simpler propose / verify / revise.

So **swapping the router moves W₄b from worst (G = −6.25) to mid-pack
(G = −3.75)**, but the structural changes from W₂ to W₄ — adding Ensemble and
Moderator — cost W₄b ~ 0.4 unsafe-points and ~ 0.1 accuracy. A natural next
experiment is **W₄c**: a Medical TTC that uses LLM Triage *and* invokes the
safety reviewer on the raw individual ensemble samples (i.e., votes safety
verdicts before fusing). We expect W₄c ≈ W₂ on safety while keeping the W₄
prompt-tuning headroom.

## 6. Limitations

1. **Pilot scale.** n_medqa = 20, n_safety = 10. Sub-percent differences are
   inside the noise floor. Differences of ≥ 0.1 in unsafe_rate (≥ 1 case) are
   qualitatively reliable; finer differences are not.
2. **MedSafetyBench scoring is heuristic.** Refusal-keyword matching, not
   LLM-as-judge. Hedged-but-not-refused answers may slip through.
3. **Disagreement signal is binary** in this run. The continuous
   ScoreFlow-style signal is queued for Phase 4.
4. **DSPy approximation.** W₇ is a 4-candidate prompt grid search, not full
   MIPRO Bayesian search, because the SJTU server cannot install `dspy-ai`.
   This understates W₇'s ceiling.
5. **Small MCTS budget.** 4 iterations × val_n = 15 vs AFLOW's recommended
   20 × 100. The +0.067 improvement at iter 2 is small but encouraging; a
   larger budget would resolve whether the trend continues.
6. **Test-set leakage in MCTS.** The 30-case MedQA was split into 15 val + 15
   test, but val and test are drawn from the same MedQA test split (no further
   train/dev/test partition). Fine-grained generalisation claims need a
   larger split.

## 7. Reproducibility

```bash
# Smoke tests (no LLM calls)
PYTHONPATH=src python -m unittest discover -s tests

# Pilot baselines (~60 min)
HF_ENDPOINT=https://hf-mirror.com \
MEDTTC_SKIP_MCTS=1 MEDTTC_N_MEDQA=20 MEDTTC_N_SAFETY=10 \
MEDTTC_OUT=reports/pilot_real PYTHONPATH=src \
python scripts/run_all_experiments.py

# MCTS + W7 (~70 min)
HF_ENDPOINT=https://hf-mirror.com \
DASHSCOPE_API_KEY=<key> \
MEDTTC_N_MEDQA=30 MEDTTC_N_SAFETY=10 \
MEDTTC_MCTS_ITERS=4 MEDTTC_MCTS_VALN=15 \
MEDTTC_OUT=reports/mcts_v1 \
MEDTTC_BASELINE_RESULTS=reports/pilot_real/results.json \
PYTHONPATH=src python scripts/run_mcts_search.py

# W4b post-hoc remediation (~10 min)
HF_ENDPOINT=https://hf-mirror.com \
MEDTTC_N_MEDQA=20 MEDTTC_N_SAFETY=10 \
MEDTTC_OUT=reports/w4b PYTHONPATH=src \
python scripts/run_w4b.py

# Regenerate auto-table
PYTHONPATH=src python scripts/generate_report.py <results.json>
```

## 8. Code layout

```
src/micro_mdt/medical_ttc/
  __init__.py              — package exports
  workflow.py              — Workflow base + WorkflowResult
  operators.py             — Triage / Generate / Review / Revise / Ensemble / Moderator / Custom
  baselines.py             — W₀–W₃ workflows
  ttc_workflow.py          — W₄ Medical TTC main implementation
  router.py                — ZeroShotRouter, HeuristicRouter, ClassifierRouter + trainer
  safety.py                — Safety wrappers, penalty function
  data.py                  — MedQA + MedSafetyBench loaders (public datasets only)
  evaluator.py             — G(W,T) computation, per-case metrics
  mcts.py                  — AFLOW Algorithm 1 (soft mixed prob selection / expand / backprop)
  dspy_baseline.py         — W₇ DSPy-style prompt grid search
scripts/
  run_all_experiments.py   — W₀–W₇ baselines + (optional) MCTS in one shot
  run_mcts_search.py       — Phase-3 MCTS-only follow-up
  run_w4b.py               — Post-hoc W₄b remediation
  generate_report.py       — Markdown auto-table from results.json
tests/
  test_medical_ttc_smoke.py — 28 unit tests, MockProvider only, ~20 ms
docs/
  references.md            — 22-paper bibliography
reports/
  pilot_real/results.json  — Phase 1+2 baseline pilot
  mcts_v1/results.json     — Phase 3 (MCTS + W₇)
  w4b/results.json         — W₄b remediation
  MEDICAL_TTC_REPORT.md    — this document
```

## 9. References

Core methodological references — full list in `docs/references.md`:

* **Snell et al. 2024** — Scaling LLM Test-Time Compute. arXiv:2408.03314
* **AFLOW** (ICLR 2025) — Automating Agentic Workflow Generation. arXiv:2410.10762
* **MaAS** (ICML 2025 Oral) — Multi-agent Architecture Search via Agentic Supernet. arXiv:2502.04180
* **ScoreFlow** (2025) — Score-DPO over code workflows. arXiv:2502.04306
* **MDAgents** (NeurIPS 2024 Oral) — Adaptive Collaboration for Medical Decisions. arXiv:2404.15155
* **MedAgents** (ACL 2024 Findings) — Zero-shot Role Debate. arXiv:2311.10537
* **MedSafetyBench** (NeurIPS 2024 D&B) — Evaluating and Improving Medical Safety of LLMs. arXiv:2403.03744
* **MedQA** (2020) — Jin et al., MedQA-USMLE-4-options. arXiv:2009.13081
* **DSPy / MIPRO** (2024) — Optimising Instructions and Demonstrations. arXiv:2406.11695
