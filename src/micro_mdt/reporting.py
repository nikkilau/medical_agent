from __future__ import annotations

from .models import MDTResult


def render_result(result: MDTResult, *, show_trace: bool = False) -> str:
    lines = [
        f"# {result.case.case_id} - {result.case.title}",
        "",
        f"- Difficulty: {result.difficulty.value}",
        f"- Status: {result.status}",
        f"- Human required: {'yes' if result.human_required else 'no'}",
        "",
        "## Final Output",
        result.final_answer,
    ]

    if result.rounds:
        lines.extend(["", "## Debate Log"])
        for round_record in result.rounds:
            lines.extend(
                [
                    f"### Round {round_record.round_index}",
                    f"Generalist:\n{round_record.proposal.content}",
                    f"Pharmacist:\n{round_record.pharmacy_review.content}",
                    f"Safety:\n{round_record.safety_review.content}",
                ]
            )

    if result.human_options:
        lines.extend(["", "## AI 决策选项（供医生参考）", result.human_options])

    if result.human_decision:
        lines.extend(["", "## Human Decision", result.human_decision])

    if result.documents:
        lines.append("")
        lines.append("## Generated Documents")
        doc_titles = {
            "followup_draft": "复诊记录草稿",
            "patient_note": "患者说明",
            "reminder": "复诊与检查提醒",
            "doctor_draft": "医生文书",
        }
        for name, content in result.documents.items():
            title = doc_titles.get(name, name)
            lines.extend([f"### {title}", content])

    if show_trace:
        lines.extend(["", "## Full Trace"])
        for item in result.trace:
            verdict = f" ({item.verdict.value})" if item.verdict else ""
            lines.extend([f"### {item.agent}{verdict}", item.content])

    return "\n".join(lines)


def render_summary(results: list[MDTResult]) -> str:
    rows = [
        "| Case | Difficulty | Status | Human Required |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        rows.append(
            f"| {result.case.case_id} | {result.difficulty.value} | "
            f"{result.status} | {'yes' if result.human_required else 'no'} |"
        )
    return "\n".join(rows)

