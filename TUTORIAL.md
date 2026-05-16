# 🎓 AI Agent 从入门到入职 —— 完整学习教程

> 跟着这份教程，你将从零开始，一步步学会构建 AI Agent。
> 不需要数学博士、不需要 GPU、不需要 985 学历。
> 只需要：一台 Windows 电脑 + 一个 API Key + 这份教程。

---

## 📋 目录 / Table of Contents

| 课时 | 内容 | 你需要多久 | 学完你能... |
|------|------|-----------|------------|
| [第 0 课](#第-0-课-环境搭建) | 环境搭建 | 30 分钟 | 装好 Python + uv + VS Code |
| [第 1 课](#第-1-课-hello-agent) | Hello Agent | 1 小时 | 写出第一个 AI 对话程序 |
| [第 2 课](#第-2-课-管道模式) | 管道模式 | 1.5 小时 | 理解 Pipeline 设计模式 |
| [第 3 课](#第-3-课-代码审查) | Code Review Agent | 2 小时 | 做一个 AI 代码审查工具 |
| [第 4 课](#第-4-课-rag-文档问答) | RAG 文档问答 | 2 小时 | 做一个能"读文件"的 AI |
| [第 5 课](#第-5-课-多-agent-协作) | 多 Agent 协作 | 2 小时 | 让 4 个 AI 像团队一样工作 |
| [第 6 课](#第-6-课-工具使用) | 工具使用 Agent | 2 小时 | 做一个能用工具的 AI |
| [第 7 课](#第-7-课-手写-transformer) | 手写 Transformer | 3 小时 | 从零实现 BERT 模型 |
| [第 8 课](#第-8-课-网页界面) | Streamlit 网页 | 1 小时 | 把你的项目做成网页 |
| [第 9 课](#第-9-课-面试准备) | 面试准备 | 1 小时 | 把项目变成工作 |

---

## 第 0 课：环境搭建

### 你需要装什么

```
你的电脑需要三样东西：
  1. Python（编程语言，就像"中文/英文"）        → python.org 下载
  2. uv（包管理器，就像"快递员"，帮你拿需要的代码包）→ pip install uv
  3. API Key（AI 服务的"钥匙"，证明你有权限调用） → DeepSeek 或 Anthropic 官网注册
```

### 一步一步来

**Step 1: 检查 Python 装了没有**

打开终端（Win+R → 输入 `cmd` → 回车），敲：

```bash
python --version
```

如果显示 `Python 3.11.x` 或更高 → ✅ 有了  
如果显示"找不到" → 去 https://python.org 下载 Python 3.11+

**Step 2: 装 uv**

```bash
pip install uv
```

验证：`uv --version` → 显示版本号就对了。

**Step 3: 克隆项目**

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
```

**Step 4: 装依赖**

```bash
uv sync
```

这一步会自动装好所有需要的 Python 包。喝杯水等 1-2 分钟。

**Step 5: 配置 API Key**

```bash
cp .env.example .env
```

然后用记事本打开 `.env` 文件，填上你的 API Key：
```
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic
```

**Step 6: 验证一切正常**

```bash
uv run python -m hello_agent.agent
```

如果看到 AI 回复了一段话 → 🎉 环境搭建成功！

---

## 第 1 课：Hello Agent

### 这节课学什么

写你的第一个 AI Agent——它问一句答一句，就像一个只会聊天的机器人。

### 先跑起来

```bash
uv run python -m hello_agent.agent
```

你会看到：
```
Demo: Single question

Q: What is an AI agent?
A: An AI agent is an autonomous system that...
```

### 代码在哪儿

打开 `hello_agent/agent.py`，你会看到这样的代码（我加了注释）：

```python
class HelloAgent(BaseAgent):
    """最简单的对话 Agent：问一句，答一句。"""

    def preprocess(self, inputs):
        # 第1步：把用户说的话包装成 API 需要的格式
        # "你好" → {"messages": [{"role": "user", "content": "你好"}], ...}
        return {
            "messages": [{"role": "user", "content": inputs}],
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
        }

    def _forward(self, model_inputs):
        # 第2步：把包装好的数据发给 AI，拿回回复
        reply = self.llm.send(**model_inputs)
        return {"reply": reply, "messages": model_inputs["messages"]}

    def postprocess(self, model_outputs):
        # 第3步：从 AI 的回复里提取纯文本
        return model_outputs["reply"]
```

### 关键概念：管道模式

所有 Agent 都遵循同一个 3 步流程——就像餐厅服务员：

```
preprocess      _forward       postprocess
（准备）    →   （调AI）   →   （格式化）
 接单              做菜            上菜
```

### 动手试试

修改 `hello_agent/config.py`，把 `system_prompt` 改成：
```python
system_prompt: str = "你是一个只会说东北话的 AI。"
```

再跑一次 `uv run python -m hello_agent.agent`。AI 会说东北话了！

---

## 第 2 课：管道模式

### 这节课学什么

理解为什么所有 Agent 都用同一个骨架——以及这个骨架从哪来的。

### 管道模式的前世今生

这个 3 步流程不是随便想的——它来自 **HuggingFace Transformers** 源码。

HuggingFace 是世界上最流行的 AI 模型库。他们的每个"任务"（文本分类、翻译、问答...）都遵循：

```python
class Pipeline:
    def run_single(self, inputs):
        model_inputs = self.preprocess(inputs)     # 文字 → 张量
        model_outputs = self._forward(model_inputs) # 模型推理
        outputs = self.postprocess(model_outputs)   # 张量 → 文字
        return outputs
```

我们的 `BaseAgent`（在 `ai_agent_playground/base.py`）就是照这个模板写的。

### 为什么这样设计？（奶奶版解释）

想象你开了一家"代写书信"的小店。

第一天只有你一个人：
- 客人说需求 → 你写成草稿 → 你誊写 → 交给客人

第二天你雇了两个人：
- 接单员（preprocess）：听客人说，填表格
- 写手（_forward）：根据表格写信
- 检查员（postprocess）：检查错别字，装信封

**三步分离的好处**：
1. 任何一步可以换人（换模型、换处理方式），不影响其他步
2. 每个人只需要知道自己这步做什么
3. 新加一种"代写英文信"服务，只需要换"写手"，其他不用动

### 核心文件

打开 `ai_agent_playground/base.py`，这就是整个项目的"宪法"——所有 Agent 都继承它。

```python
class BaseAgent(ABC):
    def run(self, inputs):
        model_inputs = self.preprocess(inputs)      # 填空1：怎么准备？
        model_outputs = self._forward(model_inputs)  # 填空2：怎么调AI？
        return self.postprocess(model_outputs)       # 填空3：怎么格式化？
```

### 动手试试

创建一个你自己的 Agent：

```python
# my_agent.py
from ai_agent_playground.base import BaseAgent

class JokeAgent(BaseAgent):
    """只会讲笑话的 Agent"""

    def preprocess(self, inputs):
        return {
            "messages": [{"role": "user", "content": f"讲一个关于{inputs}的笑话"}],
            "model": "deepseek-v4-pro[1m]",
            "max_tokens": 200,
            "system": "你是一个脱口秀演员",
        }

    def _forward(self, model_inputs):
        return {"reply": self.llm.send(**model_inputs)}

    def postprocess(self, model_outputs):
        return "😂 " + model_outputs["reply"] + " 😂"

# 试试：
# agent = JokeAgent()
# print(agent.run("程序员"))
```

---

## 第 3 课：代码审查

### 这节课学什么

做一个能审查代码的 AI——它会告诉你哪里写得不好、哪里有 bug、哪里不安全。

### 先跑起来

```bash
# 审查你自己的项目
uv run python -m code_review_agent.main .
```

输出示例：
```
## Critical Issues
- `ai_agent_playground/llm.py` — Missing env var validation
  **Line**: 31 | **Category**: bug | **Severity**: critical
  Accessing os.environ without checking existence raises KeyError
```

### 它是怎么工作的

```
你给一个目录路径
    ↓
Scanner（扫描器）→ 遍历目录，找到所有代码文件（.py, .js, .java...）
    ↓
Reviewer（审查器）→ 每个文件发给 AI，AI 找问题
    ↓
Reporter（报告器）→ 把 AI 找到的问题整理成漂亮的 Markdown 报告
```

### 关键设计：三个"手下"各司其职

打开 `code_review_agent/agent.py`：

```python
class CodeReviewAgent(BaseAgent):
    def __init__(self, config):
        self.scanner = Scanner(config)       # 手下1：找文件
        self.reviewer = Reviewer(config)     # 手下2：问AI
        self.reporter = ReportGenerator()    # 手下3：写报告
```

这是"分层组装"模式——大功能 = 小零件拼起来。

### 为什么这样设计？

如果将来你想加一个"GitHub API 扫描器"（直接从 GitHub 拉代码），只需：
1. 写一个新的 Scanner 类
2. 把 `self.scanner = Scanner(config)` 换成 `self.scanner = GitHubScanner(config)`

其他代码不用动。

### 动手试试

1. 把你的一个旧 Java 项目路径传给 code review agent
2. 看看 AI 能不能发现你没注意到的问题
3. 修改 `code_review_agent/config.py` 里的 `system_prompt`，改成中文提示词——AI 会用中文出报告

---

## 第 4 课：RAG 文档问答

### 这节课学什么

做一个能"读文档"的 AI。上传 PDF 后，向它提问，AI 会根据文档内容回答——不是凭记忆瞎编。

### 先跑起来

```bash
# 1. 喂文档
uv run python -m rag_qa_system.main ingest test_docs

# 2. 提问
uv run python -m rag_qa_system.main ask "什么是 RAG？它如何减少幻觉？"

# 3. 或者用聊天模式
uv run python -m rag_qa_system.main chat
```

### RAG 是什么？（奶奶版）

```
普通 AI 问答：
  你问："小明考试多少分？"
  AI 凭记忆回答："大概 85 吧..."（可能记错）

RAG 问答：
  你问："小明考试多少分？"
  AI 先翻成绩单 → 找到"小明：92分" → 回答："92分！（来源：成绩单第3行）"
```

RAG = **R**etrieval（检索）+ **A**ugmented（增强）+ **G**eneration（生成）

### 它是怎么工作的

```
喂文档阶段：
  PDF → 读文字 → 切成小段 → 每段转成"向量"（数学表示）→ 存到 ChromaDB

提问阶段：
  你的问题 → 转成向量 → 在 ChromaDB 里找最相似的段落 → 把段落+问题发给 AI → AI 回答
```

向量是什么？一种用数字表示的"意思"。
- "苹果"和"香蕉"的向量很近（都是水果）
- "苹果"和"汽车"的向量很远

### 关键概念：ChromaDB

ChromaDB 是一个"向量数据库"——它不是存文件，而是存"意思"。

就像一个图书馆管理员，但你不需要告诉它书名——你只需要描述"我想要一本关于...的书"，它就能帮你找到。

### 动手试试

1. 把你的一篇论文 PDF 放到 `test_docs/` 目录
2. `uv run python -m rag_qa_system.main ingest test_docs`
3. 向它提问论文里的内容
4. 试试 `sources <你的问题>` 看看检索到了什么段落

---

## 第 5 课：多 Agent 协作

### 这节课学什么

让 4 个 AI Agent 像一个开发团队一样协作——产品经理、开发者、测试、运维。

### 先跑起来

```bash
uv run python -m multi_agent_crew.main "做一个 URL 短链接服务，用 FastAPI 和 SQLite"
```

你会看到 4 个阶段依次运行：
```
Phase 1: PM Agent — Breaking down requirement
  [T-1] high   | Set up project and database
  [T-2] high   | Shorten URL endpoint (POST)
  [T-3] high   | Redirect endpoint (GET)
  [T-4] medium | Error handling and tests

Phase 2: Dev Agent — Implementing tasks
  [T-1] Set up project and database ... done (1229 chars)
  [T-2] Shorten URL endpoint (POST) ... done (2491 chars)
  ...

Phase 3: QA Agent — Reviewing code
  ...

Phase 4: DevOps Agent — Deployment config
  ...
```

### 它是怎么工作的

```
用户一句话需求
    ↓
PM Agent（产品经理）："这个需求可以拆成 4 个任务..."
    ↓
Dev Agent（开发者）：任务1→写代码，任务2→写代码，任务3→写代码...
    ↓
QA Agent（测试）："我来看看代码有没有问题..."
    ↓
DevOps Agent（运维）："这是 Dockerfile 和部署方案"
```

### 关键设计：编排而非执行

打开 `multi_agent_crew/crew.py`，`run()` 方法自己不干活——它只安排谁干什么：

```python
class Crew:
    def run(self, requirement):
        tasks = self.pm.run(requirement)        # "PM，拆任务"
        for task in tasks:
            code = self.dev.run(task)            # "Dev，写代码"
        qa_report = self.qa.run(all_code)        # "QA，检查"
        deploy = self.devops.run(all_code)       # "DevOps，部署"
        return CrewResult(...)
```

这就是"编排与实现分离"——老板不干活，只管安排。

### 为什么 4 个 Agent 能"理解"彼此？

它们不直接对话。它们通过**结构化输出格式**交流：

- PM 输出：`TASK_ID|PRIORITY|TITLE|DESCRIPTION`
- Dev 收到这个格式，能解析出"哦，我需要实现 T-1"
- QA 收到代码，知道"哦，我需要审查这些文件"

### 动手试试

1. 用你自己的项目需求跑一遍：`uv run python -m multi_agent_crew.main "做一个 xxx"`
2. 看看生成的代码质量如何
3. 试试只勾选 PM+Dev（不跑 QA 和 DevOps）

---

## 第 6 课：工具使用

### 这节课学什么

做一个能"使用工具"的 AI——不只是说，更能做。搜索网页、读写文件、执行命令、算数学题。

### 先跑起来

```bash
# 让 AI 帮你读文件并总结
uv run python -m mcp_agent.main "Read test_docs/ai_basics.txt and summarize RAG"

# 让 AI 帮你算数学
uv run python -m mcp_agent.main "Calculate sqrt(144) + 15^2"

# 让 AI 帮你写文件
uv run python -m mcp_agent.main "Write 'Hello World' to test_output.txt"
```

### ReAct 循环：AI 的"思考-行动"循环

这是整个项目最精妙的设计：

```
用户: "读 ai_basics.txt 然后总结 RAG"
    ↓
  [思考] AI想："我需要先读那个文件"
    ↓
  [行动] AI调用 read_file("test_docs/ai_basics.txt")
    ↓
  [观察] 拿到文件内容："Artificial Intelligence (AI) is intelligence..."
    ↓
  [思考] AI想："我看到了文件内容，现在总结 RAG 部分"
    ↓
  [回答] AI回复："RAG是检索增强生成，它先检索再生成，减少幻觉..."
```

这个循环最多跑 5 轮（可在 config.py 调整）。如果 AI 一直不回答，就会强制要求它给最终答案。

### 关键代码

打开 `mcp_agent/agent.py`，看 `_forward` 方法：

```python
for round_num in range(self.config.max_tool_rounds):
    reply = self.llm.send(messages=conversation)  # ① 问AI
    tool_call = self._parse_tool_call(reply)       # ② 检查是否要调工具

    if tool_call is None:
        return {"answer": reply}                   # 不调工具 → 这就是答案

    result = self.tools[tool_name](**tool_args)    # ③ 执行工具
    conversation.append({"role": "user", "content": f"Tool result: {result}"})
    # ④ 把结果发回AI，回到①
```

### 动手试试

1. 尝试让 AI 搜索网页 + 写入文件：`"Search for Python 3.13 new features and save the summary to python313.txt"`
2. 看看 AI 会不会自动先 search 再 write——两个工具串起来用
3. 添加你自己的工具：在 `mcp_agent/tools.py` 加一个函数，注册到 TOOLS 字典

---

## 第 7 课：手写 Transformer

### 这节课学什么

从零实现 BERT——ChatGPT 的"祖先"模型。用 PyTorch 一行一行写，理解 AI 大脑内部怎么工作。

### 先跑起来

```bash
# 验证模型正确性（4个测试）
uv run python -m mini_bert.verify

# 训练模型（5分钟，CPU）
uv run python -m mini_bert.train
```

### Transformer 的 7 层结构（从下往上）

```
输入文字
  ↓
[1. Embedding]  把"词"变成"数字向量"
  "苹果" → [0.2, -0.5, 0.8, ...]（768个小数）
  ↓
[2. Self-Attention]  词和词之间"交流"
  "吃"和"苹果"关系近 → attention 分数高
  "吃"和"汽车"关系远 → attention 分数低
  ↓
[3. Feed-Forward]  每个词"独立思考"
  768维 → 放大到3072维 → 压缩回768维
  ↓
[4. Transformer Block] = Attention + FFN + 残差连接
  残差是什么？"原话 + 新信息"，防止传话游戏中信息丢失
  ↓
[5. Encoder]  N个 Block 叠起来
  底层学语法 → 中层学语义 → 高层学推理
  ↓
[6. Pooler]  取 [CLS] token → 整句话的"中心思想"
  ↓
[7. Classifier]  中心思想 → 分类结果
  "这是正面评论" "这是体育新闻" ...
```

### 关键公式（面试必问）

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

Q (Query):  "我想知道什么？"——当前词发出的问题
K (Key):    "我是什么？"——每个词贴的标签
V (Value):   "我有什么信息？"——每个词携带的内容

除以 √d_k：防止点积太大导致梯度消失
Softmax：  把分数变成 0-1 之间的概率
```

打开 `mini_bert/model.py`，每一步都标注了张量形状（比如 `(32, 128, 768)`），方便你在脑子里"运行"代码。

### 验证你的实现是否正确

```bash
uv run python -m mini_bert.verify
```

4 个测试：
1. 形状测试——每层输出的维度对不对
2. Mask 测试——填充位置是否被正确忽略
3. 梯度测试——反向传播是否正常
4. 过拟合测试——模型能不能记住 16 个样本

### 动手试试

1. 改 `model.py` 里的 `num_layers` 从 4 改成 2，重新训练——参数量怎么变？速度怎么变？
2. 改 `num_heads` 从 4 改成 8——看看注意力的"头"越多越好还是越少越好
3. 试着用你自己的文本数据训练（修改 `train.py`）

---

## 第 8 课：网页界面

### 这节课学什么

用 Streamlit 把 7 个项目做成一个统一的网页——面试官打开浏览器就能试用。

### 跑起来

```bash
streamlit run app.py
```

浏览器自动打开 → 你会看到一个侧边栏，里面有 7 个 Agent 页面。

### Streamlit 是什么

一个 Python 库。你不需学 HTML/CSS/JavaScript——写 Python 就能出网页。

```python
# 这就够了！不需要写 HTML
import streamlit as st
st.title("Hello")
name = st.text_input("Your name")
st.write(f"Hi {name}!")
```

### 网页结构

```
┌─────────────────────────────────────┐
│  侧边栏         │  主区域            │
│                 │                   │
│  🏠 Home       │  ← 当前页面内容    │
│  💬 Chat       │                   │
│  📋 Review     │                   │
│  📚 RAG        │                   │
│  👥 Crew       │                   │
│  📄 Resume     │                   │
│  🔧 MCP        │                   │
│                 │                   │
└─────────────────────────────────────┘
```

### 动手试试

1. 把 Streamlit 部署到 Streamlit Cloud（免费）——简历上放链接
2. 加一个"Agent 7: 笑话生成器"页面
3. 改配色、改标题，个性化你的界面

---

## 第 9 课：面试准备

### 这节课学什么

把你的 7 个项目变成面试时的"武器"。

### 面试官可能会问的问题

| 问题 | 你的回答（参考） |
|------|---------------|
| "你做过什么 AI 项目？" | "7 个。从对话机器人到多 Agent 协作到手写 Transformer。代码都在 GitHub 上，可以演示。" |
| "什么是 RAG？" | "检索增强生成。先查文档再回答，不是凭记忆编。我的项目 3 就是一个完整的 RAG 系统。" |
| "你懂 Transformer 吗？" | "我不止懂，我还手写了一个。350 行 PyTorch，每行注释了张量形状。我可以给你讲 Q·K^T·V 的每一步。" |
| "怎么做多 Agent 协作？" | "编排模式。每个 Agent 独立工作，通过结构化输出格式协作——就像 PM 写任务单、Dev 根据任务单写代码。" |
| "设计模式？" | "我从 HuggingFace Transformers 源码提炼了 5 个模式：管道模板、配置驱动、分层组装、编排分离、共享单例。" |

### 面试时的演示流程

1. `streamlit run app.py` → 打开浏览器
2. 展示 Code Review Agent：粘贴一段有 bug 的代码，AI 找出问题
3. 展示 RAG Agent：上传你的简历 PDF，提问"我有什么技能？"
4. 展示 Resume Matcher：把面试公司的 JD 粘贴进去 → 实时分析匹配度
5. 展示 Multi-Agent Crew：现场输入一个需求，看 AI 团队协作

**面试官看到的是：可运行的代码 + 清晰的架构 + 会说人话的解释。**

### 你的简历可以写

> **AI 智能体开发**：独立完成 7 个生产级 AI Agent 项目，包括代码审查工具、RAG 知识库问答、多 Agent 协作框架、MCP 工具使用 Agent、手写 Transformer 模型。采用 HuggingFace Transformers 设计模式，200 行核心框架支撑所有项目。配套 Streamlit Web 界面可在线演示。
>
> **GitHub**: https://github.com/aidless/ai-agent-playground

---

## 🔄 学习路线图

```
第0课 环境搭建 ──────────────────────────────────────────────┐
   ↓                                                         │
第1课 Hello Agent ──→ 第2课 管道模式                          │
   ↓                      ↓                                  │
第3课 代码审查 ←── 学"分层组装"                                │
   ↓                                                         │
第4课 RAG问答 ←── 学"向量检索"                                 │
   ↓                                                         │
第5课 多Agent协作 ←── 学"编排分离"                              │
   ↓                                                         │
第6课 工具使用 ←── 学"ReAct循环"                                │
   ↓                                                         │
第7课 手写Transformer ←── 学"数学原理"                          │
   ↓                                                         │
第8课 网页界面 ←── 让面试官看到                                  │
   ↓                                                         │
第9课 面试准备 ←── 把项目变成工作 ←────────────────────────────┘
```

---

## 🆘 常见问题 / FAQ

### Q: 我是专升本，这个教程适合我吗？
**A: 这份教程就是写给你的。** 作者也是专升本。不需要名校学历，不需要数学博士。你只需要一台电脑、一个 API Key、和愿意动手的心。

### Q: API Key 怎么搞？
**A: 两种方式：**
1. DeepSeek（便宜）：去 platform.deepseek.com 注册 → 充值 10 块钱 → 拿到 API Key
2. Anthropic（更强）：去 console.anthropic.com 注册 → 拿 API Key

### Q: 需要 GPU 吗？
**A: 不需要。** 所有项目都可以在 CPU 上跑。只有第 7 课（训练 Mini-BERT）用了 PyTorch，4 层 mini 版 CPU 也能跑。

### Q: 我不会 Python 怎么办？
**A: 不需要精通。** 跟着教程跑起来，看不懂的代码先跳过——跑起来比理解更重要。跑完了再回头读注释。代码注释是老奶奶级别的。

### Q: 学完这份教程能找到工作吗？
**A: 不能保证。** 但你会拥有：
- 7 个可以演示的项目
- 一个专业的 GitHub 主页
- 面试时能聊半小时的技术深度
- 比 90% 同龄人多得多的实战经验

剩下的取决于你怎么展示。

---

## 📝 更新日志

- 2026-05-16: 初版发布（7 个项目 + 9 节课）

---

*这份教程和项目代码一样，都是活的。发现问题、有建议、想加内容？提 Issue 或 PR。*
