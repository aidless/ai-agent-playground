# 项目事实库 (facts.json)

## 项目状态
- 当前测试：56/56 全部通过
- Ollama 0.24.0 已安装，qwen2.5:7b 可运行
- CC Switch 已部署 MCP Hermes 服务
- DeepSeek API 可用（sk-44d7e...）

## 核心架构
- agent/async_core.py: 反思学习型 AsyncAgent（PLANNING→TOOL_CALL→REFLECT→LEARN→DONE）
- agent/orchestrator.py: 真多Agent编排（Crew+MessageBus+投票聚合）
- agent/governance.py: 三层治理（AuditLogger+PermissionManager+CircuitBreaker）
- agent/skills.py: 技能自创建系统（兼容 agentskills.io 标准）
- agent/context_compressor.py: 对话压缩（truncate/summarize/hybrid）
- agent/auto_memory.py: 自动记忆（每次操作自动记录）
- agent/message_bus.py: Agent间通信（direct/broadcast/delegate）
- agent/crew_agent.py: 独立Agent实例（身份/记忆/工具/ReAct循环）

## 工具链
- scripts/search_web.py: Bing搜索（中国可用）
- scripts/auto_audit.py: 项目自动审计
- scripts/self_reflect.py: 自我反思工具
- scripts/tool_runner.py: 工具容错包装（重试+回退）

## 网络环境
- 中国网络，DuckDuckGo 不可用，Bing 可用
- GitHub 被墙，需用镜像
- Bash 工具需走 Git Bash: "C:/Program Files/Git/bin/bash.exe"

## 用户（刘泽文）
- 齐鲁理工学院 2026 届 软件工程
- 求职目标：AI 应用开发岗
- 决策风格：说干就干
- 学习方式：读源码
- 回复偏好：简洁
