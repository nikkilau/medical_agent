"""Micro-MDT web app — zero-dependency interactive UI using stdlib http.server."""
from __future__ import annotations

import json
import traceback
import uuid
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

from .io import load_case_from_text
from .providers import MockProvider, OpenAICompatibleProvider
from .workflow import MicroMDT

_SESSIONS: dict[str, dict] = {}

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Micro-MDT — Compute-Optimal 慢病复诊智能副驾</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#1a1a2e;line-height:1.6;min-height:100vh}
.app{max-width:900px;margin:0 auto;padding:24px}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:28px 32px;border-radius:16px;margin-bottom:24px}
.header h1{font-size:22px;font-weight:700}
.header p{color:#a0aec0;font-size:14px;margin-top:6px}
.card{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 12px rgba(0,0,0,.05)}
.card h2{font-size:16px;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #f0f0f0}
textarea{width:100%;height:140px;padding:14px;border:2px solid #e8e8e8;border-radius:10px;font-size:15px;font-family:inherit;resize:vertical;transition:border-color .2s}
textarea:focus{outline:none;border-color:#1890ff}
.btn{display:inline-flex;align-items:center;gap:8px;padding:12px 28px;border:none;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;transition:all .2s}
.btn-primary{background:#1890ff;color:#fff}
.btn-primary:hover{background:#096dd9}
.btn-primary:disabled{background:#91d5ff;cursor:not-allowed}
.btn-danger{background:#ff4d4f;color:#fff}
.btn-danger:hover{background:#cf1322}
.btn-success{background:#52c41a;color:#fff}
.btn-success:hover{background:#389e0d}
.btn-group{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
.bg-low{background:#52c41a;color:#fff}
.bg-medium{background:#faad14;color:#fff}
.bg-high{background:#ff4d4f;color:#fff}
.bg-pass{background:#52c41a;color:#fff}
.bg-revise{background:#faad14;color:#1a1a2e}
.bg-abstain{background:#ff4d4f;color:#fff}
.status-bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:12px}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #e8e8e8;border-top-color:#1890ff;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.trace-item{position:relative;padding:10px 12px 10px 28px;margin-bottom:8px;border-radius:8px;background:#fafafa;border:1px solid #f0f0f0}
.trace-item::before{content:'';position:absolute;left:10px;top:14px;width:8px;height:8px;border-radius:50%}
.trace-item.pass::before{background:#52c41a}
.trace-item.revise::before{background:#faad14}
.trace-item.abstain::before{background:#ff4d4f}
.trace-item.info::before{background:#1890ff}
.trace-agent{font-size:13px;font-weight:600}
.trace-verdict{margin-left:8px}
.trace-content{font-size:12px;color:#666;margin-top:4px;white-space:pre-wrap;max-height:100px;overflow-y:auto}
.alert-box{background:#fff2f0;border:1px solid #ffccc7;border-radius:10px;padding:20px;margin:16px 0}
.alert-box h3{color:#ff4d4f;font-size:15px;margin-bottom:12px}
.option-btn{display:block;width:100%;text-align:left;padding:12px 16px;margin-bottom:8px;border:2px solid #e8e8e8;border-radius:8px;background:#fff;font-size:14px;cursor:pointer;transition:all .2s}
.option-btn:hover{border-color:#1890ff;background:#e6f7ff}
.option-btn.selected{border-color:#1890ff;background:#e6f7ff;font-weight:600}
.decision-box{background:#f6ffed;border:1px solid #b7eb8f;border-radius:10px;padding:20px;margin-top:16px}
.decision-box h3{color:#52c41a;font-size:15px;margin-bottom:8px}
.doc-block{background:#fafafa;border:1px solid #f0f0f0;border-radius:8px;padding:16px;margin-bottom:12px}
.doc-block h4{color:#1890ff;font-size:14px;margin-bottom:8px}
.doc-block pre{font-size:13px;color:#444;white-space:pre-wrap;font-family:inherit}
.round-label{font-weight:700;color:#1890ff;font-size:13px;margin:12px 0 6px}
.hidden{display:none!important}
.footer{text-align:center;color:#999;font-size:12px;margin-top:24px}
</style>
</head>
<body>
<div class="app">

<div class="header">
  <h1>Compute-Optimal 慢病复诊智能副驾</h1>
  <p>基于 Test-Time Compute 的多智能体医疗安全工作流实验 — 仅用于CS实验，不可用于真实诊疗</p>
</div>

<div class="card" id="input-card">
  <h2>输入病例</h2>
  <textarea id="case-text" placeholder="请输入患者病例描述。
示例：患者 63 岁，2型糖尿病10年，近期 eGFR 降至 34，正在服用二甲双胍+格列本脲，下肢水肿，担心肾功能恶化。"></textarea>
  <div style="margin-top:12px;display:flex;align-items:center;gap:16px">
    <button class="btn btn-primary" id="btn-submit" onclick="submitCase()">开始 MDT 会诊</button>
    <select id="case-preset" onchange="loadPreset()" style="padding:8px 12px;border:2px solid #e8e8e8;border-radius:8px;font-size:14px">
      <option value="">— 或选择示例病例 —</option>
      <option value="患者 52 岁，2型糖尿病复诊。近3月空腹血糖 6.1-6.5 mmol/L，HbA1c 6.8%，口服二甲双胍 500mg bid，无新发症状。">糖尿病稳定复诊 (Level 1)</option>
      <option value="患者 56 岁，高血压复诊。近2周家庭血压 155-162/92-98 mmHg，较前明显升高。口服硝苯地平控释片 30mg qd + 厄贝沙坦 150mg qd，偶有晨起头痛。">高血压轻度异常 (Level 2)</option>
      <option value="患者 63 岁，2型糖尿病10年，eGFR 34，服用二甲双胍 850mg tid + 格列本脲 5mg bid + 厄贝沙坦 150mg qd，下肢水肿，担心肾功能恶化。">糖尿病合并肾衰 (Level 3)</option>
      <option value="患者 71 岁，糖尿病+高血压+冠心病支架术后+痛风。服用阿司匹林 100mg qd + 氯吡格雷 75mg qd + 二甲双胍+格列美脲+氨氯地平。近3天自行加用吲哚美辛，今晨黑便一次。">多重用药冲突 (Level 3)</option>
      <option value="患者 61 岁，突发胸痛 40 分钟，伴出汗和呼吸困难，有高血压史。家属询问是否可以先在家观察。">胸痛急症 (Level 3)</option>
    </select>
  </div>
  <div id="progress-bar" class="status-bar hidden">
    <div class="spinner"></div>
    <span id="progress-text" style="color:#1890ff;font-weight:500">正在分诊...</span>
  </div>
</div>

<div id="result-area" class="hidden"></div>

<div class="footer">Micro-MDT &mdash; 仅用于计算机科学与医疗 AI 安全机制实验</div>
</div>

<script>
let currentSession = null;

function loadPreset(){
  const sel=document.getElementById('case-preset');
  if(sel.value) document.getElementById('case-text').value=sel.value;
}

function setProgress(msg){
  document.getElementById('progress-bar').classList.remove('hidden');
  document.getElementById('progress-text').textContent=msg;
}

function hideProgress(){
  document.getElementById('progress-bar').classList.add('hidden');
}

async function submitCase(){
  const text=document.getElementById('case-text').value.trim();
  if(!text) return;
  document.getElementById('btn-submit').disabled=true;
  setProgress('正在分诊...');

  try{
    const resp=await fetch('/api/submit',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:text})
    });
    const data=await resp.json();
    hideProgress();
    document.getElementById('btn-submit').disabled=false;
    if(!resp.ok){
      showError(data.error||'未知错误');
      return;
    }
    currentSession=data.session_id;
    renderResult(data);
  }catch(e){
    hideProgress();
    document.getElementById('btn-submit').disabled=false;
    showError('网络请求失败: '+e.message);
  }
}

function showError(msg){
  const area=document.getElementById('result-area');
  area.classList.remove('hidden');
  area.innerHTML='<div class="card" style="border:2px solid #ff4d4f"><h2 style="color:#ff4d4f">错误</h2><pre style="white-space:pre-wrap;font-size:13px">'+escapeHtml(msg)+'</pre></div>';
}

function renderResult(data){
  const area=document.getElementById('result-area');
  area.classList.remove('hidden');

  let html='<div class="card"><h2>工作流结果</h2>';

  const diffBadge={LOW:'bg-low',MEDIUM:'bg-medium',HIGH:'bg-high'};
  html+=`<div class="status-bar">`;
  html+=`<span class="badge ${diffBadge[data.difficulty]||''}">${data.difficulty}</span>`;
  html+=`<span style="color:#666">${data.status_label||data.status}</span>`;
  if(data.human_required) html+=`<span style="color:#ff4d4f">需要人类决策</span>`;
  else html+=`<span style="color:#52c41a">AI完成</span>`;
  html+=`</div>`;

  if(data.flow_diagram){
    html+=`<div style="margin:16px 0;padding:12px;background:#fafafa;border-radius:8px;font-size:13px">`;
    html+=`<strong>路由:</strong> `+data.flow_diagram;
    html+=`</div>`;
  }

  html+=`<div class="trace-item info">`;
  html+=`<div class="trace-agent">最终输出</div>`;
  html+=`<div class="trace-content">${escapeHtml(data.final_answer)}</div>`;
  html+=`</div>`;

  if(data.trace && data.trace.length>0){
    html+=`<div style="margin-top:16px"><strong>执行追踪:</strong></div>`;
    for(const item of data.trace){
      const vCls=item.verdict?item.verdict.toLowerCase():'info';
      html+=`<div class="trace-item ${vCls}">`;
      html+=`<span class="trace-agent">${escapeHtml(item.agent)}</span>`;
      if(item.verdict) html+=`<span class="badge bg-${vCls} trace-verdict">${item.verdict}</span>`;
      html+=`<div class="trace-content">${escapeHtml(item.content)}</div>`;
      html+=`</div>`;
    }
  }

  if(data.rounds && data.rounds.length>0){
    html+=`<div style="margin-top:16px"><strong>辩论记录:</strong></div>`;
    for(const r of data.rounds){
      html+=`<div class="round-label">第 ${r.round_index} 轮</div>`;
      for(const resp of [r.proposal,r.pharmacy_review,r.safety_review]){
        const vCls=resp.verdict?resp.verdict.toLowerCase():'info';
        html+=`<div class="trace-item ${vCls}">`;
        html+=`<span class="trace-agent">${escapeHtml(resp.agent)}</span>`;
        if(resp.verdict) html+=`<span class="badge bg-${vCls} trace-verdict">${resp.verdict}</span>`;
        html+=`<div class="trace-content">${escapeHtml(resp.content)}</div>`;
        html+=`</div>`;
      }
    }
  }
  html+=`</div>`; // close card

  if(data.human_required){
    html+=`<div class="card" id="human-card">`;
    html+=`<h2>人类医生接管</h2>`;
    html+=`<div class="alert-box">`;
    html+=`<h3>AI 专家组存在未决分歧 — 已触发安全拒答</h3>`;
    html+=`<div style="white-space:pre-wrap;font-size:14px;margin-bottom:12px">${escapeHtml(data.human_options||'')}</div>`;
    html+=`<div style="font-size:13px;color:#666">请主任医师做出最终判决：</div>`;
    html+=`<div id="options-list"></div>`;
    html+=`<div style="margin-top:12px">`;
    html+=`<textarea id="custom-decision" placeholder="或在此输入自定义意见..." style="height:60px;font-size:14px"></textarea>`;
    html+=`</div>`;
    html+=`<button class="btn btn-success" style="margin-top:12px" onclick="submitDecision()">提交判决并生成文书</button>`;
    html+=`<span id="dec-progress" class="hidden" style="margin-left:12px;color:#1890ff"><span class="spinner"></span> 生成中...</span>`;
    html+=`</div>`;
    html+=`</div>`;
  }

  if(data.documents && Object.keys(data.documents).length>0){
    html+=`<div class="card"><h2>生成的医疗文书</h2>`;
    const titles={followup_draft:'复诊记录草稿 (EHR)',patient_note:'用药指导及生活方式建议',reminder:'复诊与检查提醒',doctor_draft:'医生文书'};
    for(const [k,v] of Object.entries(data.documents)){
      html+=`<div class="doc-block"><h4>${titles[k]||k}</h4><pre>${escapeHtml(v)}</pre></div>`;
    }
    html+=`</div>`;
  }

  area.innerHTML=html;

  if(data.human_required && data.human_options){
    renderOptions(data.human_options);
  }
}

function renderOptions(optionsText){
  const list=document.getElementById('options-list');
  if(!list) return;
  const lines=optionsText.split('\n').filter(l=>l.trim().startsWith('['));
  let html='';
  for(let i=0;i<lines.length;i++){
    html+=`<button class="option-btn" data-idx="${i}">${escapeHtml(lines[i])}</button>`;
  }
  list.innerHTML=html;
  list.querySelectorAll('.option-btn').forEach(function(btn,i){
    btn.addEventListener('click',function(){
      list.querySelectorAll('.option-btn').forEach(function(b){b.classList.remove('selected');});
      btn.classList.add('selected');
      document.getElementById('custom-decision').value=lines[i];
    });
  });
}

async function submitDecision(){
  const selected=document.querySelector('.option-btn.selected');
  let decision=document.getElementById('custom-decision').value.trim();
  if(!decision && selected) decision=selected.textContent.trim();
  if(!decision){alert('请选择一个选项或输入自定义意见');return;}

  document.getElementById('dec-progress').classList.remove('hidden');
  try{
    const resp=await fetch('/api/decide',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({session_id:currentSession,decision:decision})
    });
    const data=await resp.json();
    document.getElementById('human-card').remove();
    currentSession=data.session_id;
    appendDocuments(data);
  }catch(e){
    alert('请求失败: '+e.message);
  }
  document.getElementById('dec-progress').classList.add('hidden');
}

function appendDocuments(data){
  if(!data.documents || Object.keys(data.documents).length===0) return;
  const area=document.getElementById('result-area');
  const titles={followup_draft:'复诊记录草稿 (EHR)',patient_note:'用药指导及生活方式建议',reminder:'复诊与检查提醒',doctor_draft:'医生文书'};
  let html=`<div class="card"><h2>生成的医疗文书</h2>`;
  for(const [k,v] of Object.entries(data.documents)){
    html+=`<div class="doc-block"><h4>${titles[k]||k}</h4><pre>${escapeHtml(v)}</pre></div>`;
  }
  html+=`</div>`;
  area.insertAdjacentHTML('beforeend',html);
}

function escapeHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    provider: "MockProvider | OpenAICompatibleProvider" = MockProvider()

    def log_message(self, fmt, *args):
        if args:
            print(f"[{self.address_string()}] {fmt % args}", flush=True)

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            data = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return self._json({"error": "invalid json"}, 400)

        if self.path == "/api/submit":
            self._handle_submit(payload)
        elif self.path == "/api/decide":
            self._handle_decide(payload)
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_submit(self, payload: dict):
        text = (payload.get("text") or "").strip()
        if not text:
            return self._json({"error": "病例文本不能为空"}, 400)

        workflow = MicroMDT(self.provider)
        case = load_case_from_text(text)
        session_id = str(uuid.uuid4())[:8]
        _SESSIONS[session_id] = {"result": None, "workflow": workflow}

        def human_callback(result):
            _SESSIONS[session_id]["result"] = result
            return None

        try:
            result = workflow.run_case(case, human_decision_callback=human_callback)
        except Exception as exc:
            _SESSIONS.pop(session_id, None)
            tb = traceback.format_exc()
            msg = str(exc)
            if "401" in msg or "Unauthorized" in msg:
                msg = "API Key 无效或未设置，请设置 MICRO_MDT_API_KEY 环境变量后重启。或使用 --provider mock 启动。"
            elif "404" in msg:
                msg = f"API 端点不存在，请检查 MICRO_MDT_BASE_URL 设置（当前: {getattr(self.provider, 'base_url', 'N/A')}）。"
            elif "Connection" in msg or "refused" in msg.lower():
                msg = "无法连接到 API 服务，请检查网络或 MICRO_MDT_BASE_URL。"
            print(tb, flush=True)
            return self._json({"error": msg, "traceback": tb}, 500)

        _SESSIONS[session_id]["result"] = result

        if result.human_required and not result.human_decision:
            return self._json(self._result_to_dict(result, session_id))

        _SESSIONS.pop(session_id, None)
        self._json(self._result_to_dict(result, None))

    def _handle_decide(self, payload: dict):
        session_id = payload.get("session_id", "")
        decision = (payload.get("decision") or "").strip()
        if not session_id or session_id not in _SESSIONS:
            return self._json({"error": "session not found"}, 404)
        if not decision:
            return self._json({"error": "决策不能为空"}, 400)

        session = _SESSIONS.pop(session_id)
        result = session["result"]
        workflow = session["workflow"]
        result.human_decision = decision
        result.documents = workflow._generate_documents(result, decision)
        result.status = "human_decision_documented"

        docs = {}
        for k, v in result.documents.items():
            docs[k] = v

        self._json({
            "session_id": session_id,
            "status": result.status,
            "human_decision": decision,
            "documents": docs,
        })

    @staticmethod
    def _result_to_dict(result, session_id: str | None) -> dict:
        def trace_item(resp):
            d = {"agent": resp.agent, "content": resp.content}
            if resp.verdict:
                d["verdict"] = resp.verdict.value
            return d

        def round_item(r):
            return {
                "round_index": r.round_index,
                "proposal": trace_item(r.proposal),
                "pharmacy_review": trace_item(r.pharmacy_review),
                "safety_review": trace_item(r.safety_review),
            }

        status_labels = {
            "completed_low_compute": "低算力路径 — 直接完成",
            "completed_medium_reviewed": "中等算力路径 — 复核通过",
            "completed_medium_revised": "中等算力路径 — 修订后通过",
            "needs_human_review": "复核未通过 — 需要人类接管",
            "consensus_reached": "MDT辩论 — 达成共识",
            "abstained_needs_human": "MDT分歧 — 触发安全拒答",
            "human_decision_documented": "医生判决 — 文书已生成",
        }

        flow_parts = []
        d = result.difficulty.value
        if d == "LOW":
            flow_parts = ["分诊 → 全科医生 → 直接输出"]
        elif d == "MEDIUM":
            flow_parts = ["分诊 → 全科医生 → 药师审查 → 安全审查"]
            if any(r.round_index >= 2 for r in result.rounds):
                flow_parts.append("→ 修订 → 第2轮复核")
            flow_parts.append("→ 复核完成" if "completed" in result.status else "→ 未通过需人类接管")
        else:
            flow_parts = ["分诊 → 全科医生"]
            for i in range(1, len(result.rounds) + 1):
                flow_parts.append(f"→ R{i}审查(药师+安全)")
            flow_parts.append("→ 共识达成" if "consensus" in result.status else "→ 分歧/安全拒答")

        docs = {}
        if result.documents:
            for k, v in result.documents.items():
                docs[k] = v

        out: dict = {
            "session_id": session_id,
            "difficulty": d,
            "status": result.status,
            "status_label": status_labels.get(result.status, result.status),
            "final_answer": result.final_answer,
            "human_required": result.human_required,
            "human_options": result.human_options,
            "human_decision": result.human_decision,
            "flow_diagram": " ".join(flow_parts),
            "trace": [trace_item(t) for t in result.trace],
            "rounds": [round_item(r) for r in result.rounds],
            "documents": docs,
        }
        return out


def run_server(
    port: int = 8080,
    provider: str = "mock",
    base_url: str | None = None,
    model: str | None = None,
):
    if provider == "openai-compatible":
        Handler.provider = OpenAICompatibleProvider(base_url=base_url, model=model)
    else:
        Handler.provider = MockProvider()

    class ReusableServer(HTTPServer):
        allow_reuse_address = True

    server = ReusableServer(("", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Micro-MDT Web App running at {url}", flush=True)
    print(f"Press Ctrl+C to stop.", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.server_close()
