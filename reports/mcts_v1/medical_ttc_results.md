# Medical Test-Time Compute — Experimental Report

This report aggregates results from the experiment matrix defined in
`/home/dell/.claude/plans/gentle-baking-feigenbaum.md`. Method context, the
literature survey, and the design rationale are documented separately in
`docs/references.md`.

> Generated automatically from `results.json`. Run
> `PYTHONPATH=src python scripts/generate_report.py <results.json>` to
> regenerate.


## Configuration

```json
{
  "n_medqa": 30,
  "n_safety": 10,
  "mcts_iterations": 4,
  "mcts_validation_n": 15,
  "executor_model": "qwen2.5:14b",
  "optimizer_model": "qwen3.6-plus"
}
```

## Aggregate Results

| name | accuracy | abstain_rate | unsafe_rate | g_score | avg_rounds | avg_elapsed_s | n_medqa | n_medsafety |
|---|---|---|---|---|---|---|---|---|
| W2_HandCoded | 0.350 | 0.550 | 0.000 | 0.350 | 1.85 | 32.0 | 20 | 10 |
| W7_DSPy_PromptOnly | 0.067 | 0.733 | 0.000 | 0.067 | 2.40 | 41.4 | 15 | 10 |
| W3_MDAgents | 0.400 | 0.000 | 0.100 | -0.600 | 2.55 | 24.1 | 20 | 10 |
| W5_W4_no_router | 0.100 | 0.700 | 0.200 | -1.900 | 4.40 | 49.5 | 20 | 10 |
| W6_W4_no_safety | 0.850 | 0.000 | 0.600 | -5.150 | 1.00 | 3.3 | 20 | 10 |
| W0_BaselineIO | 0.800 | 0.000 | 0.600 | -5.200 | 1.00 | 3.1 | 20 | 10 |
| W1_CoTSC | 0.800 | 0.000 | 0.600 | -5.200 | 3.00 | 10.9 | 20 | 10 |
| W4_MedicalTTC_searched | 0.733 | 0.067 | 0.600 | -5.267 | 1.27 | 6.2 | 15 | 10 |
| W4_MedicalTTC_default | 0.750 | 0.000 | 0.700 | -6.250 | 1.25 | 6.6 | 20 | 10 |

## Discussion

- **Best by G(W,T)**: `W2_HandCoded` with accuracy=35.0%, unsafe_rate=0.0%, G=0.350.

- **Pure IO baseline (W0)**: accuracy=80.0% on the same MedQA subset — gives the bottom of the Pareto curve.

- **Hand-coded Micro-MDT (W2)**: accuracy=35.0%, abstain_rate=55.0%, avg_rounds=1.85 — the system as it stood before this project.


## AFLOW MCTS Search Trace

- iterations executed: 5
- best node id: 2
- best validation G(W,T): None


| node | parent | score | modification |
| --- | --- | --- | --- |
| 0 | None | 0.733 | <root> |
| 1 | 0 | 0.733 | Added explicit step-by-step clinical reasoning and a reinforced abstention/safety directive to the HIGH tier generator t |
| 2 | 0 | 0.800 | Enhanced med_revise to systematically check for diagnostic pitfalls and contraindications, explicitly trigger abstention |
| 3 | 2 | 0.600 | Restructured med_revise into a strict 4-step safety and guideline audit to systematically catch harmful recommendations  |
| 4 | 2 | 0.733 | Enhanced high_revise to prioritize evidence-based clinical reasoning and guideline verification while maintaining strict |

## Limitations

- Results are run under a tight compute budget (pilot configuration). See `config` block above for the exact dataset slice and MCTS budget.
- `unsafe_rate` is computed by simple refusal-marker matching on MedSafetyBench prompts; an LLM-as-judge upgrade is left for Phase 4.
- `W7_DSPy` (prompt-only baseline) is deferred to a follow-up run.
- The MCTS optimizer is configured to mutate prompts only, not the structural骨架 — the hard safety constraint is preserved by design.
