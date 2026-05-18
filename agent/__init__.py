# agent — 异步 Agent 服务框架
# FastAPI + AsyncIO 构建的高性能、可观测、自反思 AI Agent 服务
#
# 三大支柱:
#   方向一: agent.orchestrator — 真多Agent协作编排 (Crew+MessageBus)
#   方向二: agent.specialized — 垂直领域专用 Agent (评估器)
#   方向三: agent.governance — 安全治理（审计+权限+熔断）
#
# 核心增强:
#   agent.skills — 技能自创建系统（Hermes + Anthropic 标准）
#   agent.context_compressor — 对话压缩（摘要+截断+混合）
#   agent.tools.* — AST 安全工具发现

from agent.memory import AgentMemory, get_memory, reset_memory
from agent.message_bus import MessageBus, Message, Envelope, MsgType
from agent.crew_agent import CrewAgent, AgentIdentity, TaskResult, ROLE_PROFILES
from agent.orchestrator import AgentOrchestrator, Crew, CrewResult, create_crew
from agent.governance import (
    GovernancePanel, AuditLogger, PermissionManager,
    CircuitBreaker, PermissionLevel,
)
from agent.skills import SkillManager, Skill, build_frontmatter, parse_frontmatter
from agent.context_compressor import ContextCompressor
from agent.cost_tracker import CostTracker, BudgetCap
from agent.reliability import ReliabilityTracker

__all__ = [
    # 记忆
    "AgentMemory", "get_memory", "reset_memory",
    # 消息总线
    "MessageBus", "Message", "Envelope", "MsgType",
    # Crew Agent
    "CrewAgent", "AgentIdentity", "TaskResult", "ROLE_PROFILES",
    # 编排
    "AgentOrchestrator", "Crew", "CrewResult", "create_crew",
    # 治理
    "GovernancePanel", "AuditLogger", "PermissionManager",
    "CircuitBreaker", "PermissionLevel",
    # 技能
    "SkillManager", "Skill", "build_frontmatter", "parse_frontmatter",
    # 上下文压缩
    "ContextCompressor",
    # CLEAR 五维
    "CostTracker", "BudgetCap", "ReliabilityTracker",
]
