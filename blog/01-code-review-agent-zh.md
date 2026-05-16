# 零基础手搓 AI 代码审查 Agent，200 行 Python 搞定

> 一个专升本软件工程学生的 AI 实战项目记录。没有 fancy 的框架，就是 scanner → reviewer → report 三条链路跑通。

---

## 先说背景

我今年毕业，学历不占优势，与其海投简历被拒，不如直接做作品说话。

这是我的 AI Agent 系列的第 2 个项目：**用 AI 自动审查代码质量**。输入一个项目路径，输出一份 Markdown 格式的审查报告，按严重程度分类，带代码定位。

核心代码不到 200 行，跑通只花了 2 个小时。

## 技术栈

- Python 3.11 + uv 包管理
- DeepSeek V4 Pro（走 Anthropic SDK）
- 输出：Markdown 报告

为什么选 DeepSeek？便宜，而且跟 Anthropic SDK 兼容，将来切 Claude 只需改一行。

## 三个模块，一条链路

```
code_review_agent/
├── scanner.py    → 遍历目录，收集代码文件
├── reviewer.py   → 逐文件发给 AI 审查
├── report.py     → 生成 Markdown 报告
└── main.py       → 串联
```

没上 LangChain，没上 Agent 框架。开始就要简单，东西能跑了再加复杂度。

### Scanner：找代码

遍历目录 → 按扩展名过滤（支持 Python/JS/Java/Go/Rust/SQL 等 15+ 种）→ 跳过 .git/node_modules/.venv → 200KB 上限过滤大文件 → 按语言+路径排序。

```python
CODE_EXTENSIONS = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".java": "Java", ".go": "Go", ".rs": "Rust", ".sql": "SQL",
}
```

### Reviewer：AI 大脑

核心是 prompt 设计。我踩了几个坑之后总结了三条经验：

1. **输出格式要精确约束**——"SEVERITY|LINE|CATEGORY|TITLE|DESCRIPTION" 这种管道分隔法，AI 会严格执行
2. **给例子比给指令好用**——AI 模仿例子比理解抽象规则更准
3. **不说"only report real issues"，AI 会为了显得有用而编造问题**

```python
REVIEW_PROMPT = """\
You are a senior code reviewer. For each issue:
  SEVERITY|LINE|CATEGORY|TITLE|DESCRIPTION

SEVERITY: critical | warning | info
CATEGORY: bug | security | performance | style | best-practice
Only report real issues. Skip clean files.
"""
```

### Report：出报告

解析 AI 返回的结构化数据 → 按 critical → warning → info 分组 → 生成带文件链接和行号的 Markdown。

## 第一次跑，翻了个车

用这个 Agent 审查它自己的源码，AI 报了 4 个问题：

- ❌ "anthropic 0.102.0 版本不存在" —— **胡扯，刚装的**
- ❌ "python-dotenv 1.2.2 不存在" —— **也是胡扯**
- ✅ "缺少 langchain 依赖" —— 说得对，但我故意删掉的

**所以结论是什么？AI 代码审查是第二双眼睛，不是裁判。** 它能帮你看到你漏掉的东西，但它也会幻觉。每条建议都要人工过一遍。

## 跑一下你自己项目

```bash
git clone https://github.com/aidless/ai-agent-playground.git
cd ai-agent-playground
cp .env.example .env   # 填你的 API Key
uv sync
uv run python -m code_review_agent.main /path/to/你的项目
```

## 这个系列在做什么

| # | 项目 | 状态 |
|---|------|------|
| 1 | Hello Agent — 打通 API 的第一行代码 | ✅ |
| 2 | Code Review Agent — AI 代码审查 | ✅ |
| 3 | RAG 知识库问答系统 | 下周开始 |
| 4 | 多 Agent 协作工作流 | 规划中 |

我不是大牛，只是一个正在死磕 AI 开发的应届生。如果你也在走这条路，可以关注我的 [GitHub](https://github.com/aidless) 一起进步。

---

*发在掘金，记录一个普通学生的 AI 开发之路。写给自己看，也写给同路人。*
