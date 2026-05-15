from __future__ import annotations

from .models import MDTResult

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#1a1a2e;line-height:1.6}
.container{max-width:960px;margin:0 auto;padding:24px}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:32px;border-radius:12px;margin-bottom:24px}
.header h1{font-size:24px;margin-bottom:8px}
.header .meta{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;align-items:center}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600}
.badge-low{background:#52c41a;color:#fff}
.badge-medium{background:#faad14;color:#fff}
.badge-high{background:#ff4d4f;color:#fff}
.badge-pass{background:#52c41a;color:#fff}
.badge-revise{background:#faad14;color:#1a1a2e}
.badge-abstain{background:#ff4d4f;color:#fff}
.card{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.card h2{font-size:18px;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #f0f0f0}
.flow-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;padding:8px 0}
.flow{display:flex;align-items:flex-start;gap:10px;padding:12px 4px;min-width:max-content}
.flow-node{background:#e6f7ff;border:2px solid #91d5ff;border-radius:8px;padding:10px 16px;font-size:13px;font-weight:500;text-align:center;min-width:72px;transition:all .2s}
.flow-node.active{background:#e6f7ff;border-color:#1890ff}
.flow-node.done{background:#f6ffed;border-color:#52c41a}
.flow-node.blocked{background:#fff2f0;border-color:#ff4d4f}
.flow-node.dim{opacity:.3;border-style:dashed}
.flow-arrow{color:#bfbfbf;font-size:18px;align-self:center}
.flow-arrow.active{color:#1890ff}
.flow-branch{display:flex;flex-direction:column;gap:6px}
.flow-branch-label{font-size:11px;color:#999;text-align:center}
.timeline{position:relative;padding-left:32px}
.timeline::before{content:'';position:absolute;left:12px;top:0;bottom:0;width:2px;background:#e8e8e8}
.tl-item{position:relative;margin-bottom:20px}
.tl-item::before{content:'';position:absolute;left:-24px;top:6px;width:10px;height:10px;border-radius:50%;border:2px solid #1890ff;background:#fff}
.tl-item.pass::before{border-color:#52c41a;background:#52c41a}
.tl-item.revise::before{border-color:#faad14;background:#faad14}
.tl-item.abstain::before{border-color:#ff4d4f;background:#ff4d4f}
.tl-agent{font-size:14px;font-weight:600;color:#1a1a2e}
.tl-verdict{margin-left:8px}
.tl-content{font-size:13px;color:#555;margin-top:4px;white-space:pre-wrap;background:#fafafa;padding:12px;border-radius:8px;max-height:200px;overflow-y:hidden;border:1px solid #f0f0f0;position:relative}
.tl-content.expandable{cursor:pointer}
.tl-content.overflowing::after{content:'点击展开';position:absolute;bottom:0;left:0;right:0;padding:24px 12px 8px;text-align:center;font-size:12px;color:#1890ff;background:linear-gradient(transparent,#fafafa 60%);pointer-events:none}
.tl-content.expanded{max-height:none}
.tl-content.expanded::after{display:none}
.round-header{font-weight:600;color:#1890ff;margin:16px 0 8px;font-size:15px}
.round-header:first-child{margin-top:0}
.alert{background:#fff2f0;border:1px solid #ffccc7;border-radius:8px;padding:16px;margin:16px 0}
.alert-title{color:#ff4d4f;font-weight:600;margin-bottom:8px}
.decision{background:#f6ffed;border:1px solid #b7eb8f;border-radius:8px;padding:16px;margin-top:16px}
.decision-title{color:#52c41a;font-weight:600;margin-bottom:4px}
.doc-section{margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #f0f0f0}
.doc-section:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}
.doc-section h3{font-size:15px;color:#1890ff;margin-bottom:8px}
.doc-content{font-size:13px;color:#444;white-space:pre-wrap;background:#fafafa;padding:12px;border-radius:8px}
.summary-table{width:100%;border-collapse:collapse;font-size:13px}
.summary-table th{background:#fafafa;text-align:left;padding:10px 12px;border-bottom:2px solid #e8e8e8}
.summary-table td{padding:10px 12px;border-bottom:1px solid #f0f0f0}
.disclaimer{background:#fffbe6;border:1px solid #ffe58f;border-radius:8px;padding:12px 16px;font-size:12px;color:#ad6800;margin-top:24px}
.footer{text-align:center;color:#999;font-size:12px;margin-top:24px}
@media print{.card{box-shadow:none;border:1px solid #e8e8e8}}
"""

JS = """
<script>
(function(){
  function checkOverflow(el){
    if(el.scrollHeight > el.clientHeight + 4){
      el.classList.add('overflowing')
    }
  }
  document.querySelectorAll('.tl-content.expandable').forEach(function(el){
    checkOverflow(el)
    el.addEventListener('click',function(){
      el.classList.toggle('expanded')
      el.classList.remove('overflowing')
    })
  })
  window.addEventListener('resize',function(){
    document.querySelectorAll('.tl-content.expandable:not(.expanded)').forEach(checkOverflow)
  })
})()
</script>
"""


def _difficulty_badge(d: str) -> str:
    cls = {"LOW": "badge-low", "MEDIUM": "badge-medium", "HIGH": "badge-high"}.get(d, "")
    return f'<span class="badge {cls}">{d}</span>'


def _verdict_badge(v) -> str:
    if v is None:
        return ""
    cls = {"PASS": "badge-pass", "REVISE": "badge-revise", "ABSTAIN": "badge-abstain"}.get(
        v.value, ""
    )
    return f'<span class="badge {cls}">{v.value}</span>'


def _revision_happened(result: MDTResult) -> bool:
    for item in result.trace:
        if item.agent == "GeneralistRevision":
            return True
    return any(r.round_index >= 2 for r in result.rounds)


def _consensus_reached(result: MDTResult) -> bool:
    return result.status in ("completed_medium_reviewed", "completed_medium_revised", "consensus_reached")


def _flow_diagram(result: MDTResult) -> str:
    d = result.difficulty.value
    revision_ran = _revision_happened(result)
    has_consensus = _consensus_reached(result)

    if d == "LOW":
        nodes = [
            ("Triage\n分诊", "done"),
            ("Generalist\n全科医生", "done"),
            ("输出\n建议", "done"),
        ]
    elif d == "MEDIUM":
        nodes = [
            ("Triage\n分诊", "done"),
            ("Generalist\n全科医生", "done"),
            ("Pharmacist\n药师审查", "done"),
            ("Safety\n安全伦理", "done"),
        ]
        if revision_ran:
            nodes.append(("Revision\n修订", "done"))
            nodes.append(("第2轮\n复核", "done"))
        if has_consensus:
            nodes.append(("共识达成\n输出方案", "done"))
        else:
            nodes.append(("未通过\n需人类接管", "blocked"))
    else:  # HIGH
        nodes = [
            ("Triage\n分诊", "done"),
            ("Generalist\n全科医生", "done"),
        ]
        num_rounds = len(result.rounds)
        for i in range(1, num_rounds + 1):
            nodes.append((f"R{i} 审查\n药师+安全", "done"))
            if revision_ran and i < num_rounds:
                nodes.append((f"R{i} 修订\nRevision", "done"))
        if has_consensus:
            nodes.append(("共识达成\n输出方案", "done"))
        else:
            nodes.append(("分歧/拒答\n安全拦截", "blocked"))

    if result.human_required:
        nodes.append(("人类医生\n判决", "active" if result.human_decision else "blocked"))
    if result.documents:
        nodes.append(("文书\n生成", "done"))

    parts = ['<div class="flow-wrap"><div class="flow">']
    for i, (label, style) in enumerate(nodes):
        if i > 0:
            arrow_cls = "flow-arrow active" if style != "dim" else "flow-arrow"
            parts.append(f'<span class="{arrow_cls}">→</span>')
        cls = f"flow-node {style}"
        parts.append(f'<div class="{cls}">{label}</div>')
    parts.append("</div></div>")
    return "\n".join(parts)


def _debate_timeline(result: MDTResult) -> str:
    parts: list[str] = []
    for r in result.rounds:
        parts.append(f'<div class="round-header">第 {r.round_index} 轮辩论</div>')
        for resp in [r.proposal, r.pharmacy_review, r.safety_review]:
            v = resp.verdict
            v_cls = v.value.lower() if v else ""
            parts.append(f'<div class="tl-item {v_cls}">')
            parts.append(
                f'<span class="tl-agent">{resp.agent}</span>'
                f'<span class="tl-verdict">{_verdict_badge(v)}</span>'
            )
            parts.append(f'<div class="tl-content expandable">{_escape(resp.content)}</div>')
            parts.append("</div>")
    return "\n".join(parts)


def _trace_timeline(result: MDTResult) -> str:
    parts: list[str] = []
    for item in result.trace:
        v = item.verdict
        v_cls = v.value.lower() if v else ""
        parts.append(f'<div class="tl-item {v_cls}">')
        parts.append(
            f'<span class="tl-agent">{item.agent}</span>'
            f'<span class="tl-verdict">{_verdict_badge(v)}</span>'
        )
        parts.append(f'<div class="tl-content expandable">{_escape(item.content)}</div>')
        parts.append("</div>")
    return "\n".join(parts)


def _human_section(result: MDTResult) -> str:
    parts = ['<div class="alert">']
    parts.append('<div class="alert-title">AI 专家组存在未决分歧 — 已触发安全拒答</div>')
    if result.human_options:
        parts.append(f'<div style="white-space:pre-wrap;margin-top:8px">{_escape(result.human_options)}</div>')
    parts.append("</div>")
    if result.human_decision:
        parts.append('<div class="decision">')
        parts.append('<div class="decision-title">主任医师最终判决</div>')
        parts.append(f'<div style="margin-top:4px">{_escape(result.human_decision)}</div>')
        parts.append("</div>")
    return "\n".join(parts)


def _documents_section(result: MDTResult) -> str:
    titles = {
        "followup_draft": "复诊记录草稿 (EHR)",
        "patient_note": "用药指导及生活方式建议",
        "reminder": "复诊与检查提醒",
        "doctor_draft": "医生文书",
    }
    parts: list[str] = []
    for key, content in result.documents.items():
        title = titles.get(key, key)
        parts.append('<div class="doc-section">')
        parts.append(f"<h3>{title}</h3>")
        parts.append(f'<div class="doc-content">{_escape(content)}</div>')
        parts.append("</div>")
    return "\n".join(parts)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_html(result: MDTResult) -> str:
    d = result.difficulty.value
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Micro-MDT — {result.case.case_id}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>{_escape(result.case.title)}</h1>
  <div style="color:#a0aec0">{result.case.case_id}</div>
  <div class="meta">
    {_difficulty_badge(d)}
    <span style="color:#a0aec0">Status: {result.status}</span>
    <span style="color:#a0aec0">Human required: {'yes' if result.human_required else 'no'}</span>
  </div>
</div>

<div class="card">
  <h2>工作流路由</h2>
  {_flow_diagram(result)}
</div>

<div class="card">
  <h2>病例原文</h2>
  <div class="tl-content">{_escape(result.case.text)}</div>
</div>

<div class="card">
  <h2>最终输出</h2>
  <div class="tl-content">{_escape(result.final_answer)}</div>
</div>
"""

    if result.rounds:
        html += (
            '<div class="card">'
            '<h2>辩论记录</h2>'
            f'{_debate_timeline(result)}'
            '</div>'
        )

    if result.human_required:
        html += (
            '<div class="card">'
            '<h2>人类医生接管</h2>'
            f'{_human_section(result)}'
            '</div>'
        )

    if result.documents:
        html += (
            '<div class="card">'
            '<h2>生成的医疗文书</h2>'
            f'{_documents_section(result)}'
            '</div>'
        )

    if result.trace:
        html += (
            '<div class="card">'
            '<h2>完整执行追踪</h2>'
            f'{_trace_timeline(result)}'
            '</div>'
        )

    html += (
        '<div class="disclaimer">'
        '本报告由 Micro-MDT 系统自动生成，仅用于计算机科学与医疗 AI 安全机制实验，'
        '不能作为真实诊断、处方或治疗依据。'
        '</div>'
        '<div class="footer">Micro-MDT &mdash; Compute-Optimal 慢病复诊智能副驾</div>'
        '</div>'
        f'{JS}'
        '</body></html>'
    )
    return html


def render_summary_html(results: list[MDTResult]) -> str:
    rows = []
    for i, r in enumerate(results):
        d = r.difficulty.value
        rows.append(
            "<tr>"
            f"<td>{i + 1}</td>"
            f"<td><strong>{_escape(r.case.case_id)}</strong></td>"
            f"<td>{_escape(r.case.title)}</td>"
            f"<td>{_difficulty_badge(d)}</td>"
            f"<td>{r.status}</td>"
            f"<td>{'yes' if r.human_required else 'no'}</td>"
            "</tr>"
        )

    rows_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Micro-MDT — Summary</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>Micro-MDT 运行摘要</h1>
  <div style="color:#a0aec0">总计 {len(results)} 个病例</div>
</div>
<div class="card">
  <h2>结果总览</h2>
  <table class="summary-table">
  <tr><th>#</th><th>Case ID</th><th>标题</th><th>难度</th><th>状态</th><th>需人类接管</th></tr>
  {rows_html}
  </table>
</div>
<div class="disclaimer">
  本报告由 Micro-MDT 系统自动生成，仅用于计算机科学与医疗 AI 安全机制实验。
</div>
<div class="footer">Micro-MDT &mdash; Compute-Optimal 慢病复诊智能副驾</div>
</div>
</body></html>"""
