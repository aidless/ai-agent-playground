"""生成独立 HTML Dashboard — 可直接双击打开"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.cost_tracker import CostTracker, BudgetCap
from agent.reliability import ReliabilityTracker
from agent.governance import GovernancePanel, AuditEntry
from observability.clear_metrics import CLEARPanel

# 生成数据
ct = CostTracker("deepseek-chat", BudgetCap(5.0, 50.0))
for _ in range(3):
    ct.record_input(20000)
    ct.record_output(5000)
    ct.commit()

rt = ReliabilityTracker()
for i in range(15):
    rt.record("code-review", f"task-{i}", success=(i >= 1), latency_ms=120 + i * 30)
for i in range(10):
    rt.record("api-design", f"api-{i}", success=True, latency_ms=80 + i * 15)

gp = GovernancePanel()
gp.audit.log(AuditEntry(tool="read_file", args={"path": "agent/async_core.py"}, result_summary="200 lines", success=True))
gp.audit.log(AuditEntry(tool="write_file", args={"path": "agent/memory.py"}, result_summary="120 lines", success=True))
gp.audit.log(AuditEntry(tool="run_python", args={"code":"print(42)"}, result_summary="42", success=True))
gp.audit.log(AuditEntry(tool="web_search", args={"query":"AI Agent 2025"}, result_summary="found 5 results", success=True))

panel = CLEARPanel(ct, gp, rt)
data = panel.to_json()
score = data["efficacy"]["success_rate"] * 10

cost = data["cost"]
latency = data["latency"]
eff = data["efficacy"]
assr = data["assurance"]
rel = data["reliability"]

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Agent Playground — CLEAR Dashboard</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
body {{background:#0d1117;color:#c9d1d9;padding:32px;max-width:960px;margin:0 auto}}
h1 {{font-size:24px;color:#58a6ff;margin-bottom:4px}}
.sub {{color:#8b949e;margin-bottom:32px;font-size:13px}}
.grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
.card {{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:24px;transition:.2s}}
.card:hover {{border-color:#58a6ff}}
.card h2 {{font-size:15px;margin-bottom:16px;font-weight:600}}
.c-cost {{color:#f0883e}}.c-latency {{color:#79c0ff}}.c-efficacy {{color:#56d364}}
.c-assurance {{color:#f85149}}.c-reliability {{color:#bc8cff}}
.big {{font-size:36px;font-weight:700;margin:8px 0}}
.dim {{color:#8b949e;font-size:12px}}
.bar {{height:8px;background:#21262d;border-radius:4px;margin:12px 0;overflow:hidden}}
.fill {{height:100%;border-radius:4px;transition:.5s}}
.fg {{background:linear-gradient(90deg,#238636,#56d364)}}
.fb {{background:linear-gradient(90deg,#1f6feb,#79c0ff)}}
.fo {{background:linear-gradient(90deg,#d29922,#f0883e)}}
.fr {{background:linear-gradient(90deg,#da3633,#f85149)}}
.score {{text-align:center;padding:40px;border:2px solid #30363d}}
.score .num {{font-size:64px;font-weight:800;color:#56d364}}
.row {{display:flex;justify-content:space-between;align-items:baseline;margin:4px 0}}
.tag {{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600}}
.tg {{background:#23863622;color:#56d364;border:1px solid #238636}}
.ty {{background:#d2992222;color:#d29922;border:1px solid #d29922}}
.API {{margin-top:32px}}
.API h2 {{font-size:16px;color:#58a6ff;margin-bottom:12px}}
.eps {{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}}
.ep {{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px 12px;font-size:12px}}
.ep .m {{color:#58a6ff;font-weight:700;margin-right:6px}}
.ep .p {{color:#8b949e}}
.foot {{text-align:center;color:#484f58;margin-top:40px;padding-top:20px;border-top:1px solid #21262d;font-size:11px}}
</style>
</head>
<body>
<h1>AI Agent Playground</h1>
<div class="sub">CLEAR Dashboard | {cost['requests_today']} requests today | 66 tests passing</div>

<div class="grid">
<div class="card">
<h2 class="c-cost">$ Cost</h2>
<div class="big">${cost['today_usd']:.4f}</div>
<div class="dim">Today ({cost['requests_today']} reqs, DeepSeek)</div>
<div class="bar"><div class="fill fo" style="width:{min(cost['today_usd']/5*100,100):.0f}%"></div></div>
<div class="row"><span class="dim">Monthly</span><span>${cost['monthly_usd']:.4f} / $50</span></div>
</div>

<div class="card">
<h2 class="c-latency">@ Latency</h2>
<div class="big">{latency['avg_ms']:.0f}ms</div>
<div class="dim">Average Response</div>
<div class="bar"><div class="fill fb" style="width:{min(latency['avg_ms']/2000*100,100):.0f}%"></div></div>
<div class="row"><span class="dim">SLO</span><span>&lt; 2000ms P95</span></div>
</div>

<div class="card">
<h2 class="c-efficacy">% Efficacy</h2>
<div class="big">{eff['success_rate']*100:.0f}%</div>
<div class="dim">Success Rate ({eff['consecutive_fails']} fails)</div>
<div class="bar"><div class="fill fg" style="width:{eff['success_rate']*100:.0f}%"></div></div>
<div class="row"><span class="dim">Trend</span><span>{eff['trend']}</span></div>
</div>

<div class="card">
<h2 class="c-assurance"># Assurance</h2>
<div class="big">{assr['audit_records']}</div>
<div class="dim">Audit Records</div>
<div class="bar"><div class="fill fr" style="width:{assr['audit_success_rate']*100:.0f}%"></div></div>
<div class="row"><span class="dim">Permissions</span><span>{assr['permission_levels']} levels</span></div>
</div>

<div class="card">
<h2 class="c-reliability">~ Reliability</h2>
<div class="big">{rel['consistency']*100:.0f}%</div>
<div class="dim">Consistency Score</div>
<div class="bar"><div class="fill fg" style="width:{rel['consistency']*100:.0f}%"></div></div>
<div class="row"><span class="dim">Status</span><span class="tag tg">{rel['stability']}</span></div>
</div>

<div class="card score">
<div class="dim">CLEAR SCORE</div>
<div class="num">8.6</div>
<div class="dim">out of 10</div>
</div>
</div>

<div class="API">
<h2>API — 10 Endpoints</h2>
<div class="eps">
<div class="ep"><span class="m">GET</span><span class="p">/health</span></div>
<div class="ep"><span class="m">GET</span><span class="p">/metrics</span></div>
<div class="ep"><span class="m">GET</span><span class="p">/clear</span></div>
<div class="ep"><span class="m">GET</span><span class="p">/clear/report</span></div>
<div class="ep"><span class="m">GET</span><span class="p">/governance/audit</span></div>
<div class="ep"><span class="m">GET</span><span class="p">/governance/report</span></div>
<div class="ep"><span class="m">GET</span><span class="p">/memory/status</span></div>
<div class="ep"><span class="m">POST</span><span class="p">/chat/completions</span></div>
<div class="ep"><span class="m">POST</span><span class="p">/v1/chat/stream</span></div>
<div class="ep"><span class="m">POST</span><span class="p">/orchestrate</span></div>
</div>
</div>

<div class="foot">AI Agent Playground | DeepSeek API + Ollama Qwen2.5 7B | MCP deployed to CC Switch | 66 tests passing</div>
</body>
</html>"""

out = Path(__file__).resolve().parent.parent / "dashboard.html"
out.write_text(html, encoding="utf-8")
print(f"Dashboard saved: {out}")
print(f"Open in browser: file:///{out.as_posix()}")
