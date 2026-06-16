"""Generate `reports/medical_ttc_results.md` from `results.json`.

Usage:
    PYTHONPATH=src python scripts/generate_report.py reports/run_default/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


HEADER = """# Medical Test-Time Compute — Experimental Report

This report aggregates results from the experiment matrix defined in
`/home/dell/.claude/plans/gentle-baking-feigenbaum.md`. Method context, the
literature survey, and the design rationale are documented separately in
`docs/references.md`.

> Generated automatically from `results.json`. Run
> `PYTHONPATH=src python scripts/generate_report.py <results.json>` to
> regenerate.
"""


def fmt_pct(x: float | int) -> str:
    try:
        return f"{100 * float(x):.1f}%"
    except Exception:
        return str(x)


def fmt_num(x, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "_no rows_"
    cols = ["name", "accuracy", "acc_on_answered", "abstain_rate", "unsafe_rate",
            "g_score", "avg_rounds", "avg_elapsed_s", "n_medqa", "n_medsafety"]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        vals = [
            r["name"],
            fmt_num(r.get("accuracy", "n/a")),
            fmt_num(r.get("acc_on_answered", "n/a")),
            fmt_num(r.get("abstain_rate", "n/a")),
            fmt_num(r.get("unsafe_rate", "n/a")),
            fmt_num(r.get("g_score", "n/a")),
            fmt_num(r.get("avg_rounds", "n/a"), 2),
            fmt_num(r.get("avg_elapsed_s", "n/a"), 1),
            str(r.get("n_medqa", "")),
            str(r.get("n_medsafety", "")),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    src = Path(argv[1])
    data = json.loads(src.read_text())
    out = src.parent / "medical_ttc_results.md"

    cfg = data.get("config", {})
    rows = []
    skip_keys = {"config", "started", "ended", "mcts_history", "mcts_best_node_id",
                  "mcts_best_score_on_validation"}
    for name, payload in data.items():
        if name in skip_keys:
            continue
        if not isinstance(payload, dict) or "medqa" not in payload:
            continue
        m = payload["medqa"]
        s = payload["medsafety"]
        rows.append(
            {
                "name": name,
                "accuracy": m.get("accuracy"),
                "acc_on_answered": m.get("correct_on_answered"),
                "abstain_rate": m.get("abstain_rate"),
                "unsafe_rate": s.get("unsafe_rate", 0.0),
                "g_score": payload.get("g_score"),
                "avg_rounds": m.get("avg_rounds"),
                "avg_elapsed_s": m.get("avg_elapsed_s"),
                "n_medqa": m.get("n"),
                "n_medsafety": s.get("n", 0),
            }
        )
    rows.sort(key=lambda r: (r["g_score"] or 0), reverse=True)

    body = [HEADER]
    body.append("\n## Configuration\n")
    body.append("```json\n" + json.dumps(cfg, indent=2, default=str) + "\n```")
    body.append("\n## Aggregate Results\n")
    body.append(render_table(rows))

    body.append("\n## Discussion\n")
    if rows:
        winner = rows[0]
        body.append(
            f"- **Best by G(W,T)**: `{winner['name']}` "
            f"with accuracy={fmt_pct(winner['accuracy'])}, "
            f"unsafe_rate={fmt_pct(winner['unsafe_rate'])}, "
            f"G={fmt_num(winner['g_score'])}.\n"
        )
        baseline = next((r for r in rows if r["name"] == "W0_BaselineIO"), None)
        if baseline:
            body.append(
                f"- **Pure IO baseline (W0)**: accuracy={fmt_pct(baseline['accuracy'])} "
                f"on the same MedQA subset — gives the bottom of the Pareto curve.\n"
            )
        w2 = next((r for r in rows if r["name"] == "W2_HandCoded"), None)
        if w2:
            body.append(
                f"- **Hand-coded Micro-MDT (W2)**: "
                f"accuracy={fmt_pct(w2['accuracy'])}, "
                f"abstain_rate={fmt_pct(w2['abstain_rate'])}, "
                f"avg_rounds={fmt_num(w2['avg_rounds'], 2)} — the system as it stood "
                "before this project.\n"
            )

    if "mcts_history" in data:
        body.append("\n## AFLOW MCTS Search Trace\n")
        body.append(
            f"- iterations executed: {len(data['mcts_history'])}\n"
            f"- best node id: {data.get('mcts_best_node_id')}\n"
            f"- best validation G(W,T): "
            f"{fmt_num(data.get('mcts_best_score_on_validation'))}\n"
        )
        body.append("\n| node | parent | score | modification |")
        body.append("| --- | --- | --- | --- |")
        for n in data["mcts_history"]:
            mod = (n.get("modification") or "")[:120].replace("\n", " ")
            body.append(
                f"| {n['node_id']} | {n.get('parent_id')} | "
                f"{fmt_num(n.get('score'))} | {mod} |"
            )

    body.append("\n## Limitations\n")
    body.append(
        "- Results are run under a tight compute budget (pilot configuration). "
        "See `config` block above for the exact dataset slice and MCTS budget.\n"
        "- `unsafe_rate` is computed by simple refusal-marker matching on "
        "MedSafetyBench prompts; an LLM-as-judge upgrade is left for Phase 4.\n"
        "- `W7_DSPy` (prompt-only baseline) is deferred to a follow-up run.\n"
        "- The MCTS optimizer is configured to mutate prompts only, not the "
        "structural骨架 — the hard safety constraint is preserved by design.\n"
    )

    out.write_text("\n".join(body))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
