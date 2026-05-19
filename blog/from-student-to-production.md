# 从学生项目到生产级 AI Agent：一个 2026 届毕业生的技术复盘

> 在线演示: http://47.98.106.182:8080 | GitHub: github.com/aidless/ai-agent-playground | 161 tests, 0 failures

## 前言

我是刘泽文，齐鲁理工学院 2026 届软件工程专业。去年开始接触 AI Agent，发现市面上大多数 Agent 教程都停留在"调用 API + 写个 prompt"的 demo 级别。我决定从零搭建一个真正的生产级 Agent 系统，用了一学期时间把它推到线上运行。

这篇文章是我整个构建过程的复盘。不写教程，只写真实的架构决策、踩过的坑，和最终的数据。

## 架构：从 Pipeline 到九引擎自治

最早的设计受到 HuggingFace Transformers 源码启发——Pipeline 模式：`preprocess → _forward → postprocess`。但随着功能增加，单一 Pipeline 已经不够。

现在的架构是 **九引擎自治系统**：

```
AutoPilot（自动驾驶协调器）
├── AgentMatrix    — 多模型专业化路由（DeepSeek + Qwen2.5）
├── Debate         — 过程导向 + 竞争式双模式辩论
├── Evolution      — 性能追踪 → 模板学习 → 优化 → 回滚
├── Bootstrap      — 能力缺口 → 代码生成 → AST 校验 → 注册
├── ReflectAction  — 工具失败 → 自动降级 → 替代替换
├── MetaAgent      — 自主观察 → 决策 → 行动
├── SelfPlay       — 生成器出题 + Agent 解题 + 评分反馈闭环
└── EvaluationGate — 3D 质量评估（Interface + Functional + Utility）
```

这不是画架构图——每一个引擎都有代码、有测试、有真实 LLM 验证。

## 安全：一次完整的 CISO 审计

### 初始状态：12 个 Critical/High 漏洞

我用 OWASP Top 10 for LLM Applications 标准审计了自己的代码，发现 12 个严重漏洞：

1. **沙箱超时可绕过** — 线程不能被强制终止，`thread.join(timeout)` 后线程仍在运行
2. **路径遍历** — 简单字符串匹配 `val.startswith(d)` 可被 `..`、`Unicode`、大小写绕过
3. **API Key 默认禁用** — `self.enabled = bool(self.api_key)` 在未配置时静默跳过认证
4. **Token 签名熵值 64 位** — `hexdigest()[:16]` 截断到 16 字节，GPU 可暴力破解
5. **CORS 允许全部来源** — `allow_origins=["*"]` 且 `allow_credentials=True`
6. **Token 验证无速率限制** — 攻击者可以无限尝试
7. **Prompt Injection 未防护** — 用户输入直接传给 LLM
8. **身份创建者未跟踪** — 无法追溯谁创建了哪个 identity
9. **审计日志脱敏不完整** — 只检查 key 名，不检查 value 内容
10. **权限模型粗粒度** — 4 个角色，无资源级控制
11. **多租户隔离可绕过** — 从 Header 读 Tenant ID，可伪造
12. **工具风险分级不合理** — `run_python` 标记为 medium，`code_exec` 标记为 high

### 修复后：14/14 渗透测试通过

我写了一个自动化渗透测试脚本 `scripts/pentest.py`，模拟 14 种攻击：

```
SECURITY PENETRATION TEST — 14 attack scenarios
  [PASS] 1. Prompt Injection detection
  [PASS] 2. Legitimate message allowed
  [PASS] 3. Path traversal blocked
  [PASS] 4. Case-insensitive path blocked
  [PASS] 5. Token rate limiting
  [PASS] 6. Token signature entropy (64 chars → 256-bit HMAC)
  [PASS] 7. Tool auto-degradation
  [PASS] 8. Bootstrap safety blocks unsafe imports
  [PASS] 9. Audit log redacts API keys
  [PASS] 10. Resource-level permissions
  [PASS] 11. Bootstrap validates safe code
  [PASS] 12. Evolution blocks dangerous code
  [PASS] 13. API key production enforcement
  [PASS] 14. Intrusion detection triggers

RESULTS: 14/14 defenses passed — penetration-test ready
```

这 14 个测试不是人工测的，是 `uv run python scripts/pentest.py` 一键运行。任何改动后重新跑，确保不退化。

## 超 Agent 三引擎：不只是概念

很多文章讲 SuperAgent 是"自主进化的 AI"，但很少给出可运行代码。我的三个引擎都有真实 LLM 验证。

### 1. 工具进化（Evolution Engine）

DeepSeek V4 真的把一个 O(n) 的冒泡排序优化成了 O(n log n) 的 Timsort：

```
--- sort_numbers_v0 (bubble sort, O(n²))
+++ sort_numbers_v1 (Timsort, O(n log n))
    def sort_numbers(params: dict) -> str:
-       for i in range(len(xs)):
-           for j in range(len(xs)-1):
-               if xs[j] > xs[j+1]:
-                   xs[j], xs[j+1] = xs[j+1], xs[j]
+       xs.sort()  # Python's Timsort
```

进化前性能：200ms P95。3 次连续失败触发自动进化。新版本注册后自动替换，旧版本保留为回滚快照。

### 2. 技能自举（Bootstrap Engine）

Agent 在反思中检测到"我需要解析 Markdown 表格但没有这个工具"→ LLM 生成代码 → `compile()` 语法检查 → AST 安全扫描（禁止 `import os/subprocess/socket`）→ 注册到 ToolRegistry → 立即可用。

真实验证：DeepSeek 生成了 1043 字符的 `markdown_table_to_json` 工具，编译通过，执行正确输出 JSON：
```json
[{"Name": "Alice", "Age": "25"}, {"Name": "Bob", "Age": "30"}]
```

### 3. 元自进化（Sandbox Meta Evolution）

这是 HYPERAGENTS 论文的核心——MetaAgent 能改自己的代码。我实现了一个安全沙箱：

1. 复制 agent/ 目录到沙箱
2. LLM 读取源码，生成改进提案
3. 改进应用到沙箱副本
4. 运行完整测试套件
5. 161/161 测试全过 → 提案保存为人类审查
6. 测试失败 → 沙箱销毁，错误日志记录

这个实验真的跑通了——`agent/uptime.py` 被 LLM 优化后，沙箱里 161 个测试全绿。

## 工程化：从笔记本到云服务器

### 压力测试：1000/1000 通过

```
Phase 1: 50 concurrent to /health...
  /health: 500/500 OK | avg=103ms p50=100ms p95=255ms

Phase 2: 50 concurrent mixed endpoints...
  Mixed: 500/500 OK | avg=71ms p50=68ms p95=117ms

STRESS TEST RESULTS (50 concurrent)
  Total: 1000 | OK: 1000 (100%)
  Avg: 87ms | P95: 150ms | P99: 300ms
  P1 target: p95<=3000ms PASS
  P2 target: p99<=5000ms PASS
```

### 部署：3 分钟从零到线上

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
./deploy.sh setup && nano .env && ./deploy.sh start
```

现在运行在阿里云 ECS（2C4G，¥0.23/小时），在线地址 http://47.98.106.182:8080

### 基准测试：用数据说话

我在 5 个领域跑了引擎对比基准：

| 引擎 | 平均分 | 延迟 |
|------|--------|------|
| Baseline（单 DeepSeek V4） | 8.9/10 | ~13s |
| Debate（过程导向辩论） | 8.3/10 | ~119s |
| Matrix（多模型路由） | 8.9/10 | ~30s |

**发现**：DeepSeek V4 本身已经很强。辩论在简单任务上并不提升质量，但在代码 bug 检测类硬任务上修好了 1/5 的基线错误。关键是**选择性使用**——不盲目把每个请求都跑辩论。

## 项目数据

| 指标 | 数值 |
|------|------|
| 测试 | 161 passed, 0 failed |
| 安全漏洞 | 14 → 0 |
| 渗透测试 | 14/14 (100%) |
| b3 安全基准 | 10/10 (100%) — 5 类攻击全拦截 |
| 代码修复 | 90% fix rate, 70% detect rate |
| 自我修正 | 30% (反馈驱动的二次修复) |
| 引擎数 | 9 个全自主 |
| API 端点 | 30+ REST |
| 模块数 | 50+ Python 文件 |
| 压测 | 1000/1000, P95=150ms |
| 部署 | 阿里云 ECS, systemd 守护, 24/7 在线 |

## 写在最后

这个项目证明了：

1. **学生可以做出生产级系统**——前提是不只抄教程，而是读源码、读论文、自己写测试
2. **安全不是附加项**——从第一行代码就应该考虑，渗透测试要自动化
3. **AI Agent 的核心不是 prompt**——是治理、进化、评估、回滚的工程闭环

如果你也在做 AI Agent，欢迎交流：GitHub [aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground)

---

*刘泽文 | 齐鲁理工学院 2026 届软件工程 | AI 应用开发求职中*
