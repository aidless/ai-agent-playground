# 刘泽文 — AI Agent 开发工程师

---

## 基本信息

| | | |
|---|---|---|
| **学校** | 齐鲁理工学院 软件工程 | **年级** | 2026 届本科 |
| **籍贯** | 山东泰安 | **意向** | 上海 / 杭州 |
| **电话** | — | **邮箱** | — |
| **GitHub** | [github.com/aidless](https://github.com/aidless) | **博客** | Dev.to / 掘金 |

---

## 一句话

> 学历不够，代码来凑。7 个 AI Agent 项目，全栈手写，从 Transformer 原理到 FastAPI 服务一条线打通。

---

## 技术栈

| 领域 | 技术 |
|------|------|
| **语言** | Python（主力），Java（Spring Boot） |
| **大模型** | DeepSeek API / Anthropic SDK / OpenAI 兼容接口，ReAct 循环，Tool Calling |
| **Agent 框架** | 自研 AsyncAgent（FastAPI + asyncio），LangChain，MCP 协议 |
| **后端** | FastAPI，RESTful API，SSE 流式推送，uvicorn |
| **数据库** | ChromaDB（向量），MySQL，Redis |
| **工程化** | Docker，Git，uv，Prometheus，GitHub Actions CI |
| **深度学习** | PyTorch，Transformers，LoRA 微调（PEFT） |

---

## 项目经历

### AI Agent Playground — 异步 Agent 服务框架（2026.03 - 至今）

*关键字：FastAPI · asyncio · 流式推送 · 工具系统 · Docker · Prometheus*

从零构建了一套生产级 AI Agent 服务框架，核心代码全手写：

- **异步架构**：基于 FastAPI + asyncio 实现高并发 Agent 服务，支持 SSE 流式输出边算边推
- **工具系统**：模块化 ToolRegistry + Pydantic 校验，内置 7 个工具（搜索、抓网页、计算器、文件读写、列目录、沙箱执行代码），新增工具只需新建文件
- **可观测性**：内置 trace_id 链路追踪 + Prometheus 指标（请求数、延迟、LLM/工具调用次数），每个请求可独立调试
- **状态机**：显式 Agent 状态机 `IDLE→PLANNING→TOOL_CALL→DONE/ERROR`，并发执行多工具调用
- **安全沙箱**：代码执行工具在临时目录子进程中运行，带超时控制，无法修改项目文件
- **Docker 部署**：uv 构建的多阶段 Docker 镜像，docker-compose 一键部署，带健康检查
- **API 鉴权**：可选 API Key 鉴权中间件，开发/生产双模式配置
- **技术栈**：Python，FastAPI，DeepSeek API，ChromaDB，Docker，Prometheus

### 传统 Agent 系统 — 7 个独立 Agent（2026.01 - 2026.03）

*关键字：Pipeline 模式 · Transformers 源码 · 多 Agent 协作 · RAG*

从 HuggingFace Transformers 5000+ 行源码提取 Pipeline 模式，构建了 7 个共享同一基类的 Agent：

- **BaseAgent** 基类：200 行通用骨架（preprocess → _forward → postprocess），7 个 Agent 零重复
- **Code Review Agent**：支持 15+ 编程语言的代码审查，自动检测 Bug、安全漏洞、风格问题
- **RAG Q&A**：PDF → ChromaDB 向量化 → BM25+Vector 混合检索 → 引用回答，检索准确率 > 85%
- **Multi-Agent Crew**：4 个 Agent 协作（PM→Dev→QA→DevOps），一句话需求自动产出代码
- **Resume Matcher**：简历+JD 匹配度分析，自动提取关键词对比
- **MCP Tool Agent**：基于 MCP 协议（Model Context Protocol）的动态工具调用
- **技术写作**：中英双语技术博客 3 篇（代码审查、Transformers 设计模式、多 Agent 协作）

### Mini-BERT — 从零实现 Transformer（2026.04）

*关键字：PyTorch · 多头注意力 · 位置编码 · 350 行*

用 Python + PyTorch 从零手写 BERT 模型的完整 Transformer 架构：

- 实现多头自注意力、位置编码、前馈网络、LayerNorm 等核心组件
- 每行代码标注张量形状变化（如 `[batch, seq_len, dim] → [batch, num_heads, seq_len, head_dim]`）
- 在 SST-2 文本分类任务上完成训练，准确率 > 85%

### LoRA 微调实验（2026.04）

*关键字：参数高效微调 · PEFT · 低秩适配*

- 使用 LoRA（Low-Rank Adaptation）对预训练模型进行参数高效微调
- 对比全量微调与 LoRA（r=8, r=16）的性能/效率差异
- 技术栈：Python，PyTorch，PEFT，Transformers

### Java 后端项目（2022 - 2023）

- 使用 Spring Boot + MySQL 完成企业级管理系统后端开发
- RESTful API 设计，JWT 鉴权，MyBatis 数据访问

---

## 个人优势

| 维度 | 说明 |
|------|------|
| **动手能力** | GitHub 上维护完整项目，7 个 Agent 全部可运行，有 CI/CD、Docker、监控 |
| **源码学习** | 不刷课，直接读 HuggingFace Transformers 5000+ 行源码学设计模式 |
| **技术写作** | 中英双语博客 3 篇，GitHub README 全部双语，展示技术表达能力 |
| **自驱力** | 2026 年独立完成从 Python 入门 → Agent 开发 → 服务部署的完整技术栈构建 |
| **英语** | 可流畅阅读英文技术文档、论文、源码 |

---

## 教育背景

**齐鲁理工学院** · 软件工程 · 本科 · 2026 届

---
