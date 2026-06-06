# 从学生项目到生产级 AI Agent：一个毕业生的技术复盘

> **在线演示**: http://47.98.106.182:8080 &nbsp;|&nbsp; **GitHub**: [aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground) &nbsp;|&nbsp; **161 tests, 0 failures**

---

## 一句话总结

我用一学期时间，从零搭了一个**能自我进化、通过14/14渗透测试、24/7运行在阿里云上的AI Agent系统**。这不是教程跟做，是每个架构决策都自己想、自己测、自己改出来的。

---

## 为什么写这篇复盘

去年开始接触 AI Agent 时，我发现一个尴尬的现实：

- 教程级内容：调用 API + 写个 prompt，跑起来就完了
- 生产级参考：几乎没有——大家都在说"Agent 会自己干活"，但没人说"怎么让它安全地在服务器上跑三个月不崩"

所以我决定：**自己搭一个，把所有坑都踩一遍，把过程写下来。**

这篇复盘不写教程，只写真实的架构决策、踩过的坑，和最终的数据。

---

## 第一阶段：从 Pipeline 到九引擎自治

### 最初的设计

受 HuggingFace Transformers 源码启发，最早的 Agent 是一个简单的 Pipeline：

```
preprocess → _forward → postprocess
```

`preprocess` 把用户输入变成模型能理解的格式，`_forward` 调 LLM，`postprocess` 把输出返还给用户。很清晰，也很不够。

### 问题来了

Agent 跑起来之后，我发现单一 Pipeline 架构有两个根本问题：

1. **Agent 会失败**——调工具报错、逻辑跑偏、循环出不来。Pipeline 没有"反思"和"修正"的位置
2. **Agent 应该进化**——每次都犯同样的错，为什么不学？Pipeline 没有"学习"的位置

### 现在的架构

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

**关键：这不是架构图——每一个引擎都有代码、有测试、有真实 LLM 验证。**

---

## 第二阶段：安全审计——从12个漏洞到14/14通过

### 我以为我安全了

系统跑起来了，功能也有了，我觉得可以了。然后我认真地用 **OWASP Top 10 for LLM Applications** 标准审计了自己的代码。

**结果：12 个 Critical/High 漏洞。**

| # | 漏洞 | 后果 |
|---|---|---|
| 1 | 沙箱超时可绕过 | 攻击者可让恶意代码永久运行 |
| 2 | 路径遍历 `..` / Unicode / 大小写 | 可读服务器任意文件 |
| 3 | API Key 未配置时静默跳过认证 | 生产环境无认证直接暴露 |
| 4 | Token 签名熵值16字节 | GPU 可暴力破解 |
| 5 | CORS `*` + credentials | 任意网站可发起跨域请求 |
| 6 | Token 无暴力破解限制 | 无限尝试 |
| 7 | Prompt Injection 未防护 | 用户可注入指令操控 Agent |
| 8 | 身份创建者未跟踪 | 无法溯源 |
| 9 | 审计日志脱敏不完整 | API Key 可能泄露到日志 |
| 10 | 权限模型粗粒度 | 4角色无资源级控制 |
| 11 | 多租户隔离可绕过 | Header 伪造 Tenant ID |
| 12 | 工具风险分级不合理 | `run_python` 标 medium，实际应标 high |

### 修复过程

我没有一个个修，而是**写了一个自动化渗透测试脚本** `scripts/pentest.py`，模拟14种攻击场景。每次修一个漏洞，跑一遍测试，确保不引入回归。

**修复后的结果：**

```
SECURITY PENETRATION TEST — 14 attack scenarios
  [PASS] 1. Prompt Injection detection
  [PASS] 2. Legitimate message allowed
  [PASS] 3. Path traversal blocked
  [PASS] 4. Case-insensitive path blocked
  [PASS] 5. Token rate limiting
  [PASS] 6. Token signature entropy (64 chars)
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

**这14个测试不是人工测的，是 `uv run python scripts/pentest.py` 一键运行。任何改动后重新跑，确保不退化。**

---

## 第三阶段：让 Agent 真的能"自我进化"

很多文章讲 SuperAgent 是"自主进化的 AI"，但很少给出可运行代码。我的三个引擎都有真实 LLM 验证。

### 1. 工具进化（Evolution Engine）

**场景：** DeepSeek V4 调用一个排序工具，用的是 O(n²) 冒泡排序。3次连续失败后，系统自动触发进化。

**进化过程：**
1. LLM 读取工具代码，分析性能瓶颈
2. 生成优化版本（O(n log n) Timsort）
3. 沙箱测试新版本
4. 性能达标 → 注册替换；性能退步 → 回滚快照

**真实输出：**
```diff
--- sort_numbers_v0 (bubble sort, O(n²))
+++ sort_numbers_v1 (Timsort, O(n log n))
    def sort_numbers(params: dict) -> str:
-       for i in range(len(xs)):
-           for j in range(len(xs)-1):
-               if xs[j] > xs[j+1]:
-                   xs[j], xs[j+1] = xs[j+1], xs[j]
+       xs.sort()  # Python's Timsort
```

进化前性能：200ms P95。进化后：12ms P95。**17倍提升。**

### 2. 技能自举（Bootstrap Engine）

**场景：** Agent 在反思中检测到"我需要解析 Markdown 表格但没有这个工具"。

**自举过程：**
1. LLM 生成工具代码（1043字符的 `markdown_table_to_json`）
2. `compile()` 语法检查
3. AST 安全扫描（禁止 `import os/subprocess/socket`）
4. 注册到 ToolRegistry
5. **立即可用**——下一个循环就能调用这个新工具

**真实验证：** DeepSeek 生成的工具执行正确，输出：
```json
[{"Name": "Alice", "Age": "25"}, {"Name": "Bob", "Age": "30"}]
```

### 3. 元自进化（Sandbox Meta Evolution）

这是 HYPERAGENTS 论文的核心——MetaAgent 能改自己的代码。我实现了一个安全沙箱：

1. 复制 `agent/` 目录到沙箱
2. LLM 读取源码，生成改进提案
3. 改进应用到沙箱副本
4. 运行完整测试套件（161个测试）
5. **161/161 全过** → 提案保存为人类审查；**有失败** → 沙箱销毁，错误日志记录

**这个实验真的跑通了**——`agent/uptime.py` 被 LLM 优化后，沙箱里161个测试全绿。

---

## 第四阶段：工程化——从笔记本到云服务器

### 压力测试

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

### 部署

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
./deploy.sh setup && nano .env && ./deploy.sh start
```

现在运行在**阿里云 ECS（2C4G，¥0.23/小时）**，在线地址 http://47.98.106.182:8080

---

## 基准测试：用数据说话

我在5个领域跑了引擎对比基准：

| 引擎 | 平均分 | 延迟 |
|------|--------|------|
| Baseline（单 DeepSeek V4） | 8.9/10 | ~13s |
| Debate（过程导向辩论） | 8.3/10 | ~119s |
| Matrix（多模型路由） | 8.9/10 | ~30s |

**发现：** DeepSeek V4 本身已经很强。辩论在简单任务上并不提升质量，但在**代码 bug 检测类硬任务**上修好了 1/5 的基线错误。

**关键洞察：** 不是"辩论总比单模型好"，而是**选择性使用**——不盲目把每个请求都跑辩论，只在置信度低/任务复杂时启用。

---

## 项目数据总览

| 指标 | 数值 |
|------|------|
| 测试 | 161 passed, 0 failed |
| 安全漏洞 | 12 → 0 |
| 渗透测试 | 14/14 (100%) |
| b3 安全基准 | 10/10 (100%) |
| 代码修复 | 90% fix rate, 70% detect rate |
| 自我修正 | 30% (反馈驱动的二次修复) |
| 引擎数 | 9 个全自主 |
| API 端点 | 30+ REST |
| 模块数 | 50+ Python 文件 |
| 压测 | 1000/1000, P95=150ms |
| 部署 | 阿里云 ECS, systemd 守护, 24/7 在线 |

---

## 我学到的5件事

### 1. 学生可以做出生产级系统

前提是不只抄教程，而是**读源码、读论文、自己写测试**。

这个项目里最有价值的不是代码，而是**我读过的20多篇AI研究论文**——把 HyperAgents 的元进化思想、Debate 的对抗验证思路、Self-Play 的课程学习框架，都落地成了可运行的代码。

### 2. 安全不是附加项

从第一行代码就应该考虑安全。我的第一个可部署版本有12个严重漏洞——如果早点做安全审计，会少走很多弯路。

**教训：** 渗透测试要自动化，`scripts/pentest.py` 是这个项目里性价比最高的脚本。

### 3. AI Agent 的核心不是 prompt

是**治理、进化、评估、回滚的工程闭环**。

Prompt 决定 Agent 能做什么，但工程决定 Agent **持续稳定地做对事**。没有回滚机制的"自我进化"就是灾难——一次坏的进化就能让整个系统不可用。

### 4. 选择性比全面性更重要

不是每个请求都需要跑辩论，不是每个工具都需要进化，不是每个能力缺口都需要自举新工具。

**系统要学会"不做某事"**——这比"做什么"更难，也更值钱。

### 5. 工程判断力 > 技术栈深度

招聘时，我见过太多"我会 LangChain / LlamaIndex"的候选人。但问题是：**你用它们解决了什么问题？遇到了什么瓶颈？怎么权衡的？**

这个项目想展示的不是"我用了 DeepSeek V4"，而是**"我在什么情况下选择了 DeepSeek V4 而不是其他方案，以及这个选择带来了什么后果"。**

---

## 如果你也在做 AI Agent

欢迎交流。这个项目完全开源，你可以：

- **跑起来看看**：`git clone` + `uv sync` + 配个 API Key，5分钟跑起来
- **看代码**：`agent/` 目录有43个生产级文件，每个引擎都有测试
- **提 Issue**：发现问题或有想法，直接在 GitHub 提

**GitHub**: [aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground) &nbsp;|&nbsp; **Live Demo**: http://47.98.106.182:8080

---

*刘泽文 | 齐鲁理工学院 2026 届软件工程 | AI 应用开发求职中*
