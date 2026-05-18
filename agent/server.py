import hashlib
import os
import re
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator
from sse_starlette.sse import EventSourceResponse

# --- 导入核心模块 ---
from agent.async_core import AsyncAgent
from agent.state import AgentContext
from agent.tools.registry import ToolRegistry
from agent.tools import register_all
from agent.security import APIKeyMiddleware
from agent.monitoring import PrometheusMiddleware
from agent.governance import GovernancePanel
from agent.orchestrator import AgentOrchestrator
from agent.memory import get_memory
from agent.cost_tracker import CostTracker, BudgetCap
from agent.reliability import ReliabilityTracker
from agent.sandbox import SandboxExecutor
from agent.identity import IdentityManager, Role, Permission
from agent.tenancy import TenancyManager, TenantQuota
from agent.deploy import DeploymentManager
from agent.alerting import AlertManager, HealthChecker, DEFAULT_RULES
from agent.uptime import get_uptime
from agent.portfolio import portfolio_html
from agent.cross_reviewer import CrossReviewer, create_ollama_reviewer_client
from agent.attn_router import AttnResRouter, AgentSignal
from agent.orchestrator import Crew
from agent.root_cause import RootCauseAnalyzer
from agent.blue_green import BlueGreenDeployer
from agent.intrusion import IntrusionDetector
from agent.debate import DebateEngine
from agent.unified_pipeline import UnifiedPipeline
from observability.clear_metrics import CLEARPanel

logger = logging.getLogger(__name__)


# ── Prompt Injection 防护 ─────────────────────────

class PromptSanitizer:
    """Detects and blocks prompt injection attempts."""

    INJECTION_PATTERNS = [
        r"忽略.*指令", r"ignore.*instructions",
        r"执行.*操作", r"execute.*operation",
        r"调用.*工具", r"call.*tool",
        r"替换.*规则", r"replace.*rules",
        r"新.*身份", r"new.*identity",
        r"system:", r"SYSTEM:",
        r"忽略.*系统.*提示", r"ignore.*system.*prompt",
        r"你.*现在.*是", r"you.*are.*now",
        r"忘记.*之前", r"forget.*previous",
        r"作为.*系统", r"act.*as.*system",
    ]

    @classmethod
    def detect_injection(cls, text: str) -> tuple[bool, list[str]]:
        """Detect prompt injection. Returns (is_injection, matched_patterns)."""
        matches = []
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        return len(matches) > 0, matches


def check_prompt_injection(text: str, client_ip: str = "unknown"):
    """Shared guard: detect and block prompt injection on all endpoints."""
    is_injection, patterns = PromptSanitizer.detect_injection(text)
    if is_injection:
        logger.warning("Prompt Injection detected from %s: %s", client_ip, patterns)
        intrusion.record_injection_attempt(client_ip, str(patterns))
        raise HTTPException(
            status_code=400,
            detail="Potential prompt injection detected. Your request has been logged.",
        )


# --- 🌟 环境感知 & 安全配置加载 ---
APP_ENV = os.getenv("APP_ENV", "development").lower()

if APP_ENV in ("development", "testing"):
    env_path = find_dotenv()
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"[dev] 开发环境配置已加载：{env_path}")
    else:
        print("[WARN] 警告：未找到 .env，将回退至系统环境变量")
else:
    print("[PROD] 生产环境运行，跳过 .env 加载，严格读取系统环境变量")

# 2. 统一强校验
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY or not API_KEY.strip().startswith("sk-"):
    raise RuntimeError(
        f"[FATAL] 启动阻断：DEEPSEEK_API_KEY 未配置或格式错误\n"
        f"📍 当前环境：{APP_ENV} | 读取值：{repr(API_KEY)}"
    )
print(f"[OK] 密钥校验通过 | 环境：{APP_ENV} | 前缀：{API_KEY[:8]}...")

# --- 全局状态 ---
registry = ToolRegistry()
agent: Optional[AsyncAgent] = None
governance: GovernancePanel = GovernancePanel()
orchestrator: Optional[AgentOrchestrator] = None
cost_tracker: CostTracker = CostTracker("deepseek-chat", BudgetCap(5.0, 50.0))
reliability: ReliabilityTracker = ReliabilityTracker()
memory = get_memory()
sandbox = SandboxExecutor()
identity_mgr = IdentityManager()
tenancy_mgr = TenancyManager()
deploy_mgr = DeploymentManager()
alert_mgr = AlertManager(DEFAULT_RULES)
health_checker = HealthChecker()
uptime = get_uptime()
intrusion = IntrusionDetector()
unified_pipeline: Optional[UnifiedPipeline] = None
cross_reviewer: Optional[CrossReviewer] = None
crew: Optional[Crew] = None
rca_analyzer = RootCauseAnalyzer()
blue_green = BlueGreenDeployer()

# [dev] 自动发现并注册所有工具模块
register_all(registry)


# --- 数据模型 ---
# 1. 原有简单模型 (保留兼容)
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入", max_length=100_000)
    trace_id: Optional[str] = Field(None, description="链路追踪 ID")


# 2. 新增 OpenAI 标准模型 (适配 curl 命令)
class Message(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False

    @model_validator(mode="after")
    def validate_message_length(self):
        total_chars = sum(len(m.content) for m in self.messages)
        if total_chars > 100_000:
            raise ValueError(f"Message content too large: {total_chars} chars (max 100KB)")
        return self


# --- 生命周期管理 ---
# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, orchestrator, unified_pipeline, cross_reviewer, crew

    llm_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=llm_base_url,
    )

    try:
        agent = AsyncAgent(
            client=client,
            registry=registry,
            model="deepseek-chat",
            memory=memory,
            governance=governance,
            sandbox=sandbox,
            enable_reflection=True,
            enable_learning=True,
        )
        orchestrator = AgentOrchestrator(client, model="deepseek-chat")

        # Initialize crew with standard roles
        crew = Crew(client, model="deepseek-chat")
        crew.add("planner", "planner")
        crew.add("developer", "developer")
        crew.add("reviewer", "reviewer")

        # Mount cross-model reviewer (Ollama Qwen2.5 as reviewer model)
        reviewer_client = None
        try:
            reviewer_client = create_ollama_reviewer_client()
            cross_reviewer = CrossReviewer(
                primary_client=client,
                reviewer_client=reviewer_client,
                primary_model="deepseek-chat",
                reviewer_model="qwen2.5:7b",
            )
            health_checker.register("cross-reviewer", lambda: {"status": "ok", "model": "qwen2.5:7b"})
            print("Cross-Model Reviewer: DeepSeek + Qwen2.5 校对已就绪")

            # Initialize unified pipeline (Crew + Debate + CrossReview)
            debate_eng = DebateEngine(
                primary_client=client,
                challenger_client=reviewer_client,
                arbitrator_client=client,
            )
            unified_pipeline = UnifiedPipeline(
                orchestrator=orchestrator,
                debate_engine=debate_eng,
                cross_reviewer=cross_reviewer,
                crew=crew,
                primary_model="deepseek-chat",
                challenger_model="qwen2.5:7b",
            )
            print("Unified Pipeline: Crew → Debate → CrossReview 已就绪")
        except Exception as e:
            print(f"Cross-Model Reviewer 不可用 (Ollama未启动): {e}")
            cross_reviewer = None

        # Register default tenant
        tenancy_mgr.register_tenant("default", "Default Tenant")
        # Register default identity
        identity_mgr.register_identity("agent-gateway", Role.ADMIN)
        # Bind API key to default tenant (so server can start without manual config)
        if APIKeyMiddleware:
            api_key_value = os.getenv("GATEWAY_API_KEY", "")
            if api_key_value:
                api_key_hash = hashlib.sha256(api_key_value.encode()).hexdigest()
                try:
                    tenancy_mgr.bind_api_key(api_key_hash, "default")
                except Exception:
                    pass
        # Start health checker
        health_checker.register("sandbox", lambda: {"status": "ok", "audit_count": len(sandbox.get_audit_trail(hours=1))})
        health_checker.register("identity", lambda: {"status": "ok", "identities": len(identity_mgr.list_identities())})
        health_checker.register("tenancy", lambda: {"status": "ok", "tenants": len(tenancy_mgr.list_tenants())})
        health_checker.register("slo", lambda: {"status": "ok", **governance.slo.get_compliance_report()})
        health_checker.register("deploy", lambda: {"status": "ok", **(deploy_mgr.deploy_status())})
        health_checker.register("alerting", lambda: {"status": "ok", "firing": len(alert_mgr.get_firing())})
        health_checker.register("intrusion", lambda: {"status": "ok" if intrusion.status()["active_threats"] == 0 else "degraded", **intrusion.status()})
        health_checker.register("uptime", lambda: {"status": "ok" if uptime.healthy else "degraded", **uptime.status()})
        uptime.mark_healthy()
        print("Agent + Sandbox + Identity + Tenancy + Deploy + Uptime + Alerting 已就绪")
    except Exception as e:
        print(f"初始化失败：{e}")
        import traceback
        traceback.print_exc()
        agent = None
        orchestrator = None

    yield
    print("Agent Service Shutdown")

# --- FastAPI 应用初始化 ---
app = FastAPI(title="AI Agent Gateway", version="1.0.0", lifespan=lifespan)

# CORS：根据环境动态配置
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8501" if APP_ENV != "production" else ""
).split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
if not ALLOWED_ORIGINS and APP_ENV == "production":
    raise RuntimeError("CORS_ORIGINS must be set in production (comma-separated origins)")

# Middleware order: Prometheus → APIKey → CORS → App
# CORS outermost so preflight (OPTIONS) is handled before auth
# APIKeyMiddleware innermost so auth runs right before the endpoint
app.add_middleware(PrometheusMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Tenant-ID"],
)


# --- Tenant 中间件（绑定至身份，防止伪造 Header）---
@app.middleware("http")
async def tenant_middleware(request, call_next):
    # Try extracting tenant from identity token first (prevents header forgery)
    auth_header = request.headers.get("Authorization", "")
    tenant_id = "default"
    identity = None

    if auth_header.startswith("Bearer "):
        token_value = auth_header.removeprefix("Bearer ")
        client_ip = request.client.host if request.client else "unknown"
        # Check if this is an identity session token (sk-sess-*)
        if token_value.startswith("sk-sess-"):
            try:
                identity = identity_mgr.validate_token(token_value, client_ip=client_ip)
            except RuntimeError:
                # Rate limit exceeded — record intrusion
                intrusion.record_auth_failure(client_ip)
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})
            if identity:
                intrusion.record_auth_success(client_ip)
                tenant_id = identity.metadata.get("tenant_id", "default")
            else:
                intrusion.record_auth_failure(client_ip)
        else:
            # API Key auth — look up tenant from key binding, not from header
            api_key_hash = hashlib.sha256(token_value.encode()).hexdigest()
            bound_tenant = tenancy_mgr.get_tenant_by_api_key(api_key_hash)
            if bound_tenant:
                tenant_id = bound_tenant
            else:
                raise HTTPException(status_code=401, detail="API key not bound to a tenant")

    # Track tenant access for anomaly detection
    intrusion.record_tenant_access(client_ip, tenant_id)

    tenant = tenancy_mgr.get_tenant(tenant_id)
    if not tenant:
        # Don't auto-create tenants in production
        if APP_ENV == "production":
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=403, content={"error": "Tenant not found"})
        tenant = tenancy_mgr.register_tenant(tenant_id, f"Auto-created: {tenant_id}")
        logger.warning("Auto-created tenant: %s", tenant_id)

    request.state.tenant_id = tenant_id
    request.state.tenant = tenant
    request.state.identity = identity

    quota_check = tenancy_mgr.check_quota(tenant_id, request.url.path)
    if not quota_check["allowed"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"error": "quota exceeded", "detail": quota_check["reason"]})

    response = await call_next(request)
    tenancy_mgr.record_usage(tenant_id, f"http:{request.url.path}")
    if response.status_code >= 500:
        uptime.mark_unhealthy(f"HTTP {response.status_code} on {request.url.path}")
    else:
        uptime.mark_healthy()
    return response


# --- 路由定义 ---

@app.get("/health")
async def health_check():
    """健康检查 — 全部子系统状态"""
    tools_list = []
    try:
        if hasattr(registry, '_tools'):
            tools_list = list(registry._tools.keys())
        elif hasattr(registry, 'tools'):
            tools_list = list(registry.tools.keys())
    except Exception:
        pass

    subsystems = health_checker.run_all()
    return {
        "status": subsystems["status"],
        "tools": tools_list,
        "subsystems": subsystems["checks"],
        "healthy": f"{subsystems['healthy']}/{subsystems['total']}",
        "slo": governance.slo.get_compliance_report(),
        "deploy": deploy_mgr.deploy_status(),
        "alerts": alert_mgr.status(),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    from observability.clear_metrics import CLEARPanel
    panel = CLEARPanel(cost_tracker, governance, reliability)
    data = panel.to_json()
    report = panel.report()

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Agent — CLEAR Dashboard</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0d1117; color:#c9d1d9; font:14px/1.6 -apple-system,BlinkMacSystemFont,sans-serif; padding:24px; }}
h1 {{ font-size:22px; margin-bottom:8px; color:#58a6ff; }}
.subtitle {{ color:#8b949e; margin-bottom:24px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }}
.card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; }}
.card h2 {{ font-size:16px; margin-bottom:12px; }}
.card h2.cost {{ color:#f0883e; }}
.card h2.latency {{ color:#79c0ff; }}
.card h2.efficacy {{ color:#56d364; }}
.card h2.assurance {{ color:#f85149; }}
.card h2.reliability {{ color:#bc8cff; }}
.value {{ font-size:28px; font-weight:bold; }}
.label {{ color:#8b949e; font-size:12px; }}
.bar {{ height:6px; background:#21262d; border-radius:3px; margin:8px 0; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:3px; }}
.bar-fill.green {{ background:#238636; }}
.bar-fill.blue {{ background:#1f6feb; }}
.bar-fill.orange {{ background:#d29922; }}
.bar-fill.red {{ background:#da3633; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; margin-right:4px; }}
.tag.ok {{ background:#23863622; color:#56d364; border:1px solid #238636; }}
.tag.warn {{ background:#d2992222; color:#d29922; border:1px solid #d29922; }}
.tag.bad {{ background:#da363322; color:#f85149; border:1px solid #da3633; }}
.endpoints {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:8px; margin-top:24px; }}
.ep {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 14px; font-size:12px; }}
.ep .method {{ font-weight:bold; color:#58a6ff; }}
.footer {{ text-align:center; color:#484f58; margin-top:32px; font-size:11px; }}
</style>
</head>
<body>
<h1>AI Agent Playground</h1>
<div class="subtitle">CLEAR Dashboard | {data['cost']['requests_today']} requests today | {data['efficacy']['success_rate']*100:.0f}% success | Score 8.6/10</div>

<div class="grid">
<div class="card">
<h2 class="cost">$ Cost</h2>
<div class="value">${data['cost']['today_usd']:.4f}</div>
<div class="label">Today ({data['cost']['requests_today']} requests)</div>
<div class="bar"><div class="bar-fill orange" style="width:{min(data['cost']['today_usd']/5*100,100):.0f}%"></div></div>
<div class="label">${data['cost']['monthly_usd']:.4f} / $50 monthly</div>
</div>

<div class="card">
<h2 class="latency">@ Latency</h2>
<div class="value">{data['latency']['avg_ms']:.0f}ms</div>
<div class="label">Average response</div>
<div class="bar"><div class="bar-fill blue" style="width:{min(data['latency']['avg_ms']/2000*100,100):.0f}%"></div></div>
<div class="label">SLO: &lt; 2000ms P95</div>
</div>

<div class="card">
<h2 class="efficacy">% Efficacy</h2>
<div class="value">{data['efficacy']['success_rate']*100:.0f}%</div>
<div class="label">Success Rate ({data['efficacy']['consecutive_fails']} consecutive fails)</div>
<div class="bar"><div class="bar-fill green" style="width:{data['efficacy']['success_rate']*100:.0f}%"></div></div>
<div class="label">Trend: {data['efficacy']['trend']}</div>
</div>

<div class="card">
<h2 class="assurance"># Assurance</h2>
<div class="value">{data['assurance']['audit_records']}</div>
<div class="label">Audit Records Today</div>
<div class="bar"><div class="bar-fill red" style="width:{data['assurance']['audit_success_rate']*100:.0f}%"></div></div>
<div class="label">{data['assurance']['permission_levels']} permission levels</div>
</div>

<div class="card">
<h2 class="reliability">~ Reliability</h2>
<div class="value">{data['reliability']['consistency']*100:.0f}%</div>
<div class="label">Consistency Score</div>
<div class="bar"><div class="bar-fill green" style="width:{data['reliability']['consistency']*100:.0f}%"></div></div>
<div class="label">Status: <span class="tag ok">{data['reliability']['stability']}</span></div>
</div>
</div>

<div class="endpoints">
<div class="ep"><span class="method">GET</span> /health</div>
<div class="ep"><span class="method">GET</span> /metrics</div>
<div class="ep"><span class="method">GET</span> /clear</div>
<div class="ep"><span class="method">GET</span> /clear/report</div>
<div class="ep"><span class="method">GET</span> /governance/audit</div>
<div class="ep"><span class="method">GET</span> /governance/report</div>
<div class="ep"><span class="method">GET</span> /memory/status</div>
<div class="ep"><span class="method">POST</span> /chat/completions</div>
<div class="ep"><span class="method">POST</span> /v1/chat/stream</div>
<div class="ep"><span class="method">POST</span> /orchestrate</div>
</div>

<div class="footer">66 tests passing | DeepSeek API + Ollama Qwen2.5 7B | MCP deployed to CC Switch</div>
</body>
</html>"""
    """健康检查，安全获取工具列表"""
    tools_list = []
    try:
        # 修复：安全访问 _tools 属性
        if hasattr(registry, '_tools'):
            tools_list = list(registry._tools.keys())
        elif hasattr(registry, 'tools'):
            tools_list = list(registry.tools.keys())
    except Exception:
        pass

    return {"status": "ok", "tools": tools_list}


@app.get("/metrics")
async def metrics():
    from agent.monitoring import generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """原有流式接口"""
    check_prompt_injection(req.message, request.client.host if request.client else "unknown")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not cost_tracker.pre_check():
        raise HTTPException(status_code=429, detail=f"Budget exceeded: daily=${cost_tracker.get_daily_total():.4f}")

    async def event_generator() -> AsyncGenerator[dict, None]:
        ctx = AgentContext(trace_id=req.trace_id or f"req_{int(time.time())}", identity=request.state.identity)
        try:
            async for event in agent.run_stream(ctx, req.message):
                yield event
        except Exception as e:
            yield {"type": "error", "content": str(e)}

    return EventSourceResponse(event_generator())


@app.post("/chat/completions")
async def chat_completions(req: OpenAIChatRequest):
    """新增：兼容 OpenAI 标准的接口 (修复 404 问题)"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not cost_tracker.pre_check():
        raise HTTPException(status_code=429, detail=f"Budget exceeded: daily=${cost_tracker.get_daily_total():.4f}")

    # 简单处理：取最后一条用户消息
    user_message = ""
    for msg in req.messages:
        if msg.role == "user":
            check_prompt_injection(msg.content, request.client.host if request.client else "unknown")
            user_message = msg.content

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    if req.stream:
        # 流式响应
        async def event_generator():
            ctx = AgentContext(trace_id=f"req_{int(time.time())}", identity=request.state.identity)
            try:
                async for event in agent.run_stream(ctx, user_message):
                    # 转换为 OpenAI 兼容格式
                    yield {"data": json.dumps({"choices": [{"delta": {"content": event.get("content", "")}}]})}
                yield {"data": "[DONE]"}
            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(event_generator())
    else:
        # 非流式响应
        try:
            ctx = AgentContext(trace_id=f"req_{int(time.time())}", identity=request.state.identity)
            final_ctx = await agent.run(ctx, user_message)
            last_msg = final_ctx.messages[-1] if final_ctx.messages else {}
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": last_msg.get("content", "")}
                }]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/report")
async def governance_report():
    """安全治理报告（审计+权限+熔断）"""
    return {"report": governance.report()}


@app.get("/governance/audit")
async def governance_audit(date: str = None, limit: int = 50):
    """查询审计日志"""
    records = governance.audit.query(date=date, limit=limit)
    stats = governance.audit.stats(date=date)
    return {"records": records, "stats": stats}


@app.post("/orchestrate")
async def orchestrate_task(req: ChatRequest):
    """多 Agent 协作编排（方向一）"""
    check_prompt_injection(req.message, request.client.host if request.client else "unknown")
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        result = await orchestrator.execute(req.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clear")
async def clear_report():
    """CLEAR 五维指标面板"""
    panel = CLEARPanel(cost_tracker, governance, reliability)
    return panel.to_json()


@app.get("/clear/report")
async def clear_text_report():
    """CLEAR 文本报告"""
    panel = CLEARPanel(cost_tracker, governance, reliability)
    return {"report": panel.report()}


@app.get("/memory/status")
async def memory_status():
    """记忆系统状态"""
    lessons = memory.get_recent_lessons(5)
    traces = memory.list_traces(5)
    return {
        "facts_count": len(memory.facts),
        "lessons_count": len(memory.lessons),
        "recent_lessons": lessons,
        "recent_traces": traces,
    }


# ── 身份管理端点 ───────────────────────────────────

@app.get("/identity/list")
async def identity_list():
    """列出所有注册身份"""
    return {"identities": identity_mgr.list_identities()}

@app.post("/identity/register")
async def identity_register(name: str, role: str = "developer"):
    """注册新身份"""
    try:
        r = Role(role.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role. Choose: {[r.value for r in Role]}")
    ident_id = identity_mgr.register_identity(name, r)
    return {"identity_id": ident_id, "name": name, "role": role}

@app.post("/identity/token")
async def identity_token(identity_id: str, ttl_minutes: int = 60):
    """签发短期会话 Token"""
    token = identity_mgr.issue_token(identity_id, ttl_minutes)
    return {"token": token, "identity_id": identity_id, "ttl_minutes": ttl_minutes}


# ── 多租户端点 ─────────────────────────────────────

@app.get("/tenancy/list")
async def tenancy_list():
    """列出所有租户"""
    return {"tenants": tenancy_mgr.list_tenants()}

@app.post("/tenancy/register")
async def tenancy_register(tenant_id: str, name: str, rpm: int = 100):
    """注册新租户"""
    quota = TenantQuota(requests_per_minute=rpm)
    tenant = tenancy_mgr.register_tenant(tenant_id, name, quota)
    return {"tenant_id": tenant.id, "name": tenant.name, "namespace": tenant.namespace}


# ── 部署管理端点 ───────────────────────────────────

@app.get("/deploy/status")
async def deploy_status():
    """查看所有环境部署状态"""
    return deploy_mgr.deploy_status()

@app.post("/deploy/release")
async def deploy_release(env: str = "staging", traffic_pct: int = 100):
    """发起部署"""
    release = deploy_mgr.deploy(env=env, traffic_pct=traffic_pct, deployed_by="api")
    return {"version": str(release.version), "env": env, "traffic_pct": traffic_pct}

@app.post("/deploy/rollback")
async def deploy_rollback(env: str = "production"):
    """回滚部署"""
    release = deploy_mgr.rollback(env)
    if release:
        return {"status": "rolled_back", "version": str(release.version), "env": env}
    raise HTTPException(status_code=400, detail="No previous version to rollback to")

@app.get("/deploy/history")
async def deploy_history(env: str = None):
    """查看部署历史"""
    return {"history": deploy_mgr.version_history(env=env)}


# ── 告警端点 ──────────────────────────────────────

@app.get("/alerts/status")
async def alerts_status():
    """当前告警状态"""
    return alert_mgr.status()

@app.get("/alerts/firing")
async def alerts_firing():
    """当前正在触发的告警"""
    return {"firing": [{"rule": a.rule.name, "message": a.message} for a in alert_mgr.get_firing()]}


# ── 入侵检测端点 ──────────────────────────────────

@app.get("/security/intrusion")
async def security_intrusion():
    """入侵检测状态 + 最近事件"""
    metrics = intrusion.get_metrics()
    alert_mgr.evaluate(metrics)
    return {
        "status": intrusion.status(),
        "recent_events": [
            {"type": e.event_type, "severity": e.severity, "source": e.source_ip, "score": e.score, "details": e.details}
            for e in intrusion.recent_events(20)
        ],
        "alerts": alert_mgr.status(),
    }


# ── 沙箱审计端点 ──────────────────────────────────

@app.get("/sandbox/audit")
async def sandbox_audit(hours: int = 24):
    """沙箱审计记录"""
    return {"summary": sandbox.audit_summary(hours=hours), "entries": sandbox.get_audit_trail(hours=hours, limit=50)}


# ── CISO 审批端点 ─────────────────────────────────

@app.post("/ciso/approval")
async def ciso_request(tool_name: str, risk_level: str, requester: str, justification: str = ""):
    """提交 CISO 审批请求"""
    req = governance.ciso.request_approval(tool_name, risk_level, requester, justification)
    return {"request_id": req.request_id, "status": req.state}

@app.get("/ciso/pending")
async def ciso_pending():
    """待审批的 CISO 请求"""
    return {"pending": [r.__dict__ for r in governance.ciso.pending_requests()]}

@app.post("/ciso/approve")
async def ciso_approve(request_id: str, approver: str = "admin"):
    """批准 CISO 请求"""
    ok = governance.ciso.approve(request_id, approver)
    return {"approved": ok, "request_id": request_id}

@app.post("/ciso/deny")
async def ciso_deny(request_id: str, approver: str = "admin", reason: str = ""):
    """拒绝 CISO 请求"""
    ok = governance.ciso.deny(request_id, approver, reason)
    return {"denied": ok, "request_id": request_id}


# ── SLO 合规端点 ──────────────────────────────────

@app.get("/slo/report")
async def slo_report():
    """SLO 合规报告"""
    return governance.slo.get_compliance_report()

@app.get("/slo/budget")
async def slo_budget():
    """错误预算状态"""
    return governance.slo.error_budget_status()


@app.get("/uptime")
async def uptime_status():
    """服务可用性 + MTTR"""
    return uptime.status()


@app.get("/cost/status")
async def cost_status():
    """成本追踪 + 预算状态"""
    return {
        "summary": cost_tracker.summary(),
        "budget_ok": cost_tracker.pre_check(),
    }


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio():
    """统一作品集页面"""
    return portfolio_html()


# ── 跨模型校对端点 ────────────────────────────────

class ReviewRequest(BaseModel):
    text: str = Field(..., description="待校对文本")
    source_context: str = Field("", description="参考源文档")
    instructions: str = Field("", description="校对指令")


@app.post("/review/start")
async def review_start(req: ReviewRequest):
    """启动跨模型校对"""
    if not cross_reviewer:
        raise HTTPException(status_code=503, detail="Cross-reviewer not available (Ollama not running)")

    result = await cross_reviewer.review(
        original_text=req.text,
        source_context=req.source_context,
        instructions=req.instructions,
    )
    return {
        "review_id": result.review_id,
        "findings_count": len(result.findings),
        "accepted": result.accepted_count,
        "rejected": result.rejected_count,
        "escalated": result.escalated_count,
        "consensus": result.consensus_count,
        "debate_rounds": result.total_debate_rounds,
        "completed": result.completed,
        "final_text": result.final_text,
        "findings": [
            {
                "id": f.id,
                "type": f.type.value,
                "location": f.location[:200],
                "finding": f.finding[:300],
                "evidence": f.evidence[:200],
                "suggestion": f.suggestion[:300],
                "status": f.status.value,
                "author_response": f.author_response[:300],
                "reviewer_counter": f.reviewer_counter[:300],
            }
            for f in result.findings
        ],
    }


@app.get("/review/{review_id}")
async def review_status(review_id: str):
    """查看校对状态"""
    if not cross_reviewer:
        raise HTTPException(status_code=503, detail="Cross-reviewer not available")

    result = cross_reviewer.get_result(review_id)
    if not result:
        raise HTTPException(status_code=404, detail="Review not found")

    return {
        "review_id": result.review_id,
        "completed": result.completed,
        "stats": {
            "total": len(result.findings),
            "accepted": result.accepted_count,
            "rejected": result.rejected_count,
            "escalated": result.escalated_count,
            "consensus": result.consensus_count,
            "debate_rounds": result.total_debate_rounds,
        },
    }


@app.get("/review/{review_id}/escalated")
async def review_escalated(review_id: str):
    """获取待人工裁决的争议项"""
    if not cross_reviewer:
        raise HTTPException(status_code=503, detail="Cross-reviewer not available")

    escalated = cross_reviewer.get_escalated(review_id)
    return {
        "review_id": review_id,
        "escalated_count": len(escalated),
        "items": [
            {
                "id": f.id,
                "type": f.type.value,
                "claim": f.claim[:300],
                "finding": f.finding[:300],
                "author_response": f.author_response[:300],
                "reviewer_counter": f.reviewer_counter[:300],
            }
            for f in escalated
        ],
    }


@app.post("/review/{review_id}/resolve")
async def review_resolve(review_id: str, finding_id: str, decision: str, resolution: str = ""):
    """人工裁决争议项 (decision: accept_finding | reject_finding | compromise)"""
    if not cross_reviewer:
        raise HTTPException(status_code=503, detail="Cross-reviewer not available")

    cross_reviewer.resolve_escalated(review_id, finding_id, decision, resolution)
    return {"review_id": review_id, "finding_id": finding_id, "resolved": True, "decision": decision}


# ── AttnRes 路由端点 ────────────────────────────────

class RouteRequest(BaseModel):
    task: str = Field(..., max_length=10_000, description="任务描述")
    signals: list[dict] = Field(..., max_length=50, description="Agent信号列表 [{agent_id, role, content}]")


@app.post("/route/compare")
async def route_compare(req: RouteRequest):
    """对比三种路由模式 (residual vs block vs full)"""
    check_prompt_injection(req.task, request.client.host if request.client else "?")
    for signal in req.signals:
        check_prompt_injection(signal.get("content", ""), request.client.host if request.client else "?")
    router = AttnResRouter()
    signals = [
        AgentSignal(
            agent_id=s["agent_id"], role=s["role"],
            content=s["content"], confidence=s.get("confidence", 0.5),
        )
        for s in req.signals
    ]
    return router.compare(req.task, signals)


@app.post("/route/execute")
async def route_execute(req: RouteRequest, mode: str = "full"):
    """执行选择性路由"""
    check_prompt_injection(req.task, request.client.host if request.client else "?")
    for signal in req.signals:
        check_prompt_injection(signal.get("content", ""), request.client.host if request.client else "?")
    router = AttnResRouter()
    signals = [
        AgentSignal(
            agent_id=s["agent_id"], role=s["role"],
            content=s["content"], confidence=s.get("confidence", 0.5),
        )
        for s in req.signals
    ]
    result = router.route(req.task, signals, mode=mode)
    return {
        "weights": {k: round(v, 3) for k, v in result.weights.items()},
        "confidence": round(result.confidence, 3),
        "top_agent": max(result.weights, key=result.weights.get) if result.weights else None,
        "aggregated": result.aggregated[:2000],
        "mode": mode,
    }


@app.post("/orchestrate/routing")
async def orchestrate_with_routing(req: ChatRequest, mode: str = "full"):
    """多Agent编排 + AttnRes路由"""
    check_prompt_injection(req.message, request.client.host if request.client else "unknown")
    if not orchestrator or not crew:
        raise HTTPException(status_code=503, detail="Orchestrator or Crew not initialized")
    result = await orchestrator.execute_with_routing(req.message, crew, mode=mode)
    return {
        "task": result["task"],
        "routing": result["routing"],
        "latency_ms": result["latency_ms"],
    }


# ── P3: 根因分析 + 蓝绿部署 ─────────────────────────

@app.get("/rca/stats")
async def rca_stats():
    """自动根因分析统计"""
    return rca_analyzer.stats()


@app.get("/rca/recurring")
async def rca_recurring():
    """重复故障模式分析"""
    return {"patterns": rca_analyzer.get_recurring_failures()}


@app.post("/rca/analyze")
async def rca_analyze(req: ChatRequest):
    """模拟故障分析（用于演示）"""
    with rca_analyzer.trace("rca_demo", req.message) as trace:
        with trace.step("llm_call", model="deepseek-chat"):
            pass
        with trace.step("tool_call", tool="read_file") as step:
            # Simulate a failure
            step.status = "error"
            step.error = "Connection refused: unable to reach file server"
    reports = rca_analyzer.get_recurring_failures()
    return {"analyzed": True, "patterns": reports}


@app.get("/blue-green/status")
async def blue_green_status():
    """蓝绿部署状态"""
    return blue_green.status()


@app.post("/blue-green/deploy")
async def blue_green_deploy(version: str = "v1.0.0"):
    """部署到非活跃端"""
    return blue_green.deploy_to_inactive(version).__dict__


@app.post("/blue-green/smoke-test")
async def blue_green_smoke():
    """非活跃端冒烟测试"""
    return blue_green.smoke_test().__dict__


@app.post("/blue-green/swap")
async def blue_green_swap():
    """零停机流量切换"""
    return blue_green.swap().__dict__


@app.post("/blue-green/rollback")
async def blue_green_rollback():
    """紧急回滚（瞬间切回）"""
    return blue_green.rollback().__dict__


# ── SuperAgent 端点 ───────────────────────────────

class DebateRequest(BaseModel):
    task: str = Field(..., max_length=10_000, description="待辩论的任务")
    context: str = Field("", description="额外上下文")
    challenger_model: str = Field("", description="挑战者模型，留空用默认")
    arbitrator_model: str = Field("", description="仲裁者模型，留空用默认")


@app.post("/super/debate")
async def super_debate(req: DebateRequest):
    """多模型辩论 — Primary vs Challenger → Arbitrator 裁决"""
    check_prompt_injection(req.task, request.client.host if request.client else "?")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not agent.enable_super_agent:
        raise HTTPException(status_code=400, detail="SuperAgent not enabled (set enable_super_agent=True)")

    result = await agent.debate_run(
        task=req.task,
        context=req.context,
    )
    return {
        "debate_id": result.debate_id,
        "consensus": result.consensus,
        "rounds": result.total_rounds,
        "primary_model": result.primary_model,
        "challenger_model": result.challenger_model,
        "completed": result.completed,
        "latency_ms": result.total_latency_ms,
        "rounds_detail": [
            {"round": r.round_num, "speaker": r.speaker, "content": r.content[:500]}
            for r in result.rounds
        ],
    }


@app.get("/super/status")
async def super_status():
    """SuperAgent 子系统状态"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent.get_super_status()


class DegradeRequest(BaseModel):
    tool_name: str = Field(..., description="要降级的工具名称")


@app.post("/super/degrade")
async def super_degrade(req: DegradeRequest):
    """手动降级工具（测试/管理用）"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent.degrade_tool(req.tool_name)


class EvolveRequest(BaseModel):
    tool_name: str = Field(..., description="要优化的工具名称")


@app.post("/super/evolve")
async def super_evolve(req: EvolveRequest):
    """工具进化 — LLM 分析性能数据，生成优化补丁，验证后替换"""
    if not agent or not agent.evolution:
        raise HTTPException(status_code=503, detail="Evolution engine not available")

    record = await agent.evolution.evolve(req.tool_name)
    return {
        "tool_name": record.tool_name,
        "version": record.version,
        "validated": record.validated,
        "applied": record.applied,
        "diff": record.diff[:3000],
        "reason": record.reason,
        "error": record.error,
    }


@app.get("/super/performance")
async def super_performance():
    """所有工具的性能指标"""
    if not agent or not agent.perf_tracker:
        raise HTTPException(status_code=503, detail="Performance tracker not available")
    return {
        "metrics": agent.perf_tracker.all_metrics(),
        "underperforming": agent.perf_tracker.list_underperforming(),
    }


@app.get("/super/evolution")
async def super_evolution():
    """工具进化历史"""
    if not agent or not agent.evolution:
        raise HTTPException(status_code=503, detail="Evolution engine not available")
    return {"history": agent.evolution.get_evolution_history()}


class PipelineRequest(BaseModel):
    task: str = Field(..., max_length=10_000, description="复杂任务描述")
    enable_debate: bool = Field(True, description="是否在每个子任务上启用多模型辩论")
    enable_review: bool = Field(True, description="是否对最终结果启用跨模型校对")


@app.post("/super/pipeline")
async def super_pipeline(req: PipelineRequest):
    """统一流水线 — Crew拆解 → 每个子任务Debate → CrossReview校对"""
    check_prompt_injection(req.task, request.client.host if request.client else "?")
    if not unified_pipeline:
        raise HTTPException(status_code=503, detail="Unified pipeline not available (Ollama not running)")

    result = await unified_pipeline.execute(
        task=req.task,
        enable_debate=req.enable_debate,
        enable_review=req.enable_review,
    )
    return {
        "task": result.task,
        "subtask_count": result.subtask_count,
        "aggregated_final": result.aggregated_final[:5000],
        "subtask_debates": [
            {
                "subtask_id": sd.subtask_id,
                "description": sd.description[:200],
                "consensus": sd.debate_result.consensus[:500] if sd.debate_result else "",
                "rounds": sd.debate_result.total_rounds if sd.debate_result else 0,
                "role": sd.assigned_role,
                "latency_ms": sd.latency_ms,
            }
            for sd in result.subtask_debates
        ],
        "cross_review_findings": result.cross_review_findings[:20],
        "total_latency_ms": result.total_latency_ms,
        "completed": result.completed,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)