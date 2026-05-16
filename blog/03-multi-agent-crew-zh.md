# 我用 4 个 AI Agent 搭了一个虚拟开发团队，一句话需求进去，完整项目出来

> 一个专升本学生的第 4 个 AI 项目。PM → Dev → QA → DevOps，四个 Agent 协作，全部基于自己写的 Pipeline 框架。

---

## 这个项目干什么

你说一句话，四个 AI Agent 协作输出一个完整项目：

```
输入："用 FastAPI 做一个 URL 短链接服务"
          │
    ┌─────▼──────┐
    │  PM Agent   │  → 拆成 4 个开发任务
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │  Dev Agent  │  → 每个任务写出完整代码
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │  QA Agent   │  → 审查代码：bug、安全、边界
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │DevOps Agent │  → Dockerfile + docker-compose + 部署清单
    └────────────┘
```

4 个 API 调用，一个 orchestrator 串联。没用 LangChain，没用 CrewAI，就是自己写的 `BaseAgent` Pipeline 模式。

## 怎么做到的

### 四个 Agent，本质上是一样的

所有 Agent 都继承自同一个 `BaseAgent`：

```python
class ProductManagerAgent(BaseAgent):
    # 唯一的区别：system_prompt
```

PM Agent 的 prompt："你是资深产品经理，把需求拆成具体任务。"
Dev Agent 的 prompt："你是资深开发者，写出可运行的代码。"
QA Agent 的 prompt："你是测试工程师，审查代码质量。"
DevOps Agent 的 prompt："你是运维工程师，生成部署方案。"

**同一个类，四个不同的 system prompt，就是四个不同的 Agent。**

### 编排器：把流程串起来

```python
class Crew:
    def run(self, requirement: str) -> CrewResult:
        tasks = self.pm.run(requirement)        # Phase 1
        for task in tasks:
            code = self.dev.run(task)            # Phase 2
        qa_report = self.qa.run(all_code)        # Phase 3
        deploy = self.devops.run(all_code)       # Phase 4
        return CrewResult(...)
```

这就是从 Transformers 源码里学的——**编排和实现分离**。`run()` 只负责"谁在什么时候干什么"，每个 Agent 自己干自己的活。

### 结构化输出是关键

PM 的输出格式是严格约束的：

```
TASK_ID|PRIORITY|TITLE|DESCRIPTION
```

用管道符分隔，程序可以直接解析。没有这个，Agent 之间的数据传递就断了。

## 实际跑一次

需求："用 FastAPI 和 SQLite 做一个 URL 短链接服务"

**PM 输出：**
- T-1 (high): 项目搭建 + 数据库初始化
- T-2 (high): POST /shorten 短链接生成
- T-3 (high): GET /{code} 重定向
- T-4 (medium): 错误处理 + pytest 测试

**Dev 输出：4 个完整的 Python 文件，总计约 9000 字符的可运行代码**

**QA 输出：代码审查报告**

**DevOps 输出：Dockerfile + docker-compose.yml**

## 四个项目的完整架构

```
ai-agent-playground/
├── ai_agent_playground/         ← 核心框架（3 个文件，200 行）
│   ├── config.py                ← BaseAgentConfig
│   ├── base.py                  ← BaseAgent Pipeline
│   └── llm.py                   ← LLMClient 单例
│
├── hello_agent/                 ← 项目 1：Hello Agent
├── code_review_agent/           ← 项目 2：AI 代码审查
├── rag_qa_system/               ← 项目 3：RAG 知识库问答
├── multi_agent_crew/            ← 项目 4：多 Agent 协作
│
└── blog/                        ← 技术博客
```

核心框架只有 200 行，支撑了 4 个完全不同类型的 Agent 项目。这就是好架构的价值。

## 学到的四条经验

**1. Prompt 就是架构**

四个 Agent 用同一个 `BaseAgent` 类，区别只在 prompt。Prompt 定义了 Agent 的行为边界、输出格式和质量标准。

**2. 结构化输出让 Agent 可以对话**

PM 输出 `TASK_ID|PRIORITY|TITLE|DESCRIPTION`，Dev 才能解析并逐个实现。这是 Agent 之间的"API 协议"。

**3. 顺序流已经足够有用**

没有做 Agent 之间的辩论循环，没有做 self-reflection。就是 PM → Dev → QA → DevOps 一条线。对 demo 来说够用，看着也清楚。

**4. 共享 LLM 客户端**

四个 Agent 共用一个 `LLMClient` 实例。一个连接池，一次 .env 加载。不浪费。

## 跑一下

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env   # 填 API Key
uv sync
uv run python -m multi_agent_crew.main "做一个 xxx"
```

---

*这是我 AI 作品集的最后一个项目。四个项目全部开源。下一步：把作品变成工作。*

*GitHub: [github.com/aidless/ai-agent-playground](https://github.com/aidless/ai-agent-playground)*
