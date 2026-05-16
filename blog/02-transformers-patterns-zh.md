# 我啃了 5000 行 HuggingFace Transformers 源码，提炼出 5 个设计模式，重构了自己的 AI Agent 项目

> 一个专升本学生的源码阅读笔记。不聊虚的，直接看代码。

---

## 前情提要

我在做一个 AI Agent 作品集，已经写了两个 Agent：一个聊天机器人，一个代码审查工具。功能都跑通了，但代码越写越难受：

- 配置散落在四五个文件里，改个参数要找半天
- 每个 Agent 自己创建 Anthropic 客户端，重复代码一堆
- hello_agent 和 code_review_agent 之间零复用
- 看不出整体架构

于是我做了一件事：**读 HuggingFace Transformers 源码**。

不是读文档，不是看教程，是读源码。`modeling_utils.py` 5091 行、`generation/utils.py` 3935 行、`pipelines/base.py` 1373 行。

读完我提取了 5 个设计模式，全部落地到了自己的项目里。这篇文章就是记录这个过程。

---

## 模式 1：管道模板方法

**Transformers 怎么做：**

每个任务——文本生成、分类、翻译——都是同一套流程：

```python
class Pipeline:
    def run_single(self, inputs, ...):
        model_inputs = self.preprocess(inputs, ...)    # 文字 → 张量
        model_outputs = self.forward(model_inputs, ...) # 模型推理
        outputs = self.postprocess(model_outputs, ...)  # 张量 → 文字
```

**我怎么做：**

```python
class BaseAgent(ABC):
    def run(self, inputs, **kwargs):
        model_inputs = self.preprocess(inputs, **kwargs)
        model_outputs = self._forward(model_inputs, **kwargs)
        return self.postprocess(model_outputs, **kwargs)
```

现在每新增一个 Agent，只需实现 3 个方法。就这么简单。

---

## 模式 2：配置驱动

**Transformers 怎么做：**

BERT 的配置只有 67 行：

```python
class BertConfig(PreTrainedConfig):
    model_type = "bert"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
```

1365 行的 `PreTrainedConfig` 基类干了所有的活（save / load / 序列化 / 验证）。子类只声明"我有哪些参数、默认值是什么"。

**我怎么做：**

```python
@dataclass
class BaseAgentConfig:
    model: str = "deepseek-v4-pro[1m]"
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful AI assistant."

@dataclass
class CodeReviewConfig(BaseAgentConfig):
    agent_type = "code-review"
    max_file_bytes: int = 200_000
    max_files_per_run: int = 30
    skip_dirs: set = field(default_factory=lambda: {".git", ".venv", ...})
```

**重构前**：改一个参数，要在多个文件里找。
**重构后**：改一个 dataclass 字段，一行搞定。

---

## 模式 3：分层组装

**Transformers 怎么做：**

BERT 不是一个大类，而是小零件拼出来的：

```
BertEmbeddings → BertEncoder → BertPooler
                    ↓
              12 × BertLayer
                    ↓
        BertAttention + BertIntermediate + BertOutput
```

每个类只做一件事。`BertSelfAttention` 只算注意力，`BertSelfOutput` 只管残差连接，`BertLayer` 把它们拼起来。

**我怎么做：**

```python
class CodeReviewAgent(BaseAgent):
    def __init__(self, config):
        self.scanner = Scanner(config)       # 文件系统 → FileInfo
        self.reviewer = Reviewer(config)     # FileInfo → AI 审查 → Issues
        self.reporter = ReportGenerator()    # Issues → Markdown 报告
```

每个组件可以独立测试。将来想把本地 scanner 换成 GitHub API scanner？改一行。

---

## 模式 4：编排与实现分离

**Transformers 怎么做：**

`generate()` 方法 200 行，但它自己不干活——它编排 9 个 helper：

```
generate()
  ├── 1. 解析参数，确定生成模式
  ├── 2. 初始化 logits 处理器
  ├── 3. 准备模型输入
  ├── ...
  └── 9. 调用具体解码方法（贪婪/采样/束搜索）
```

真正的 greedy search 逻辑在别的方法里。`generate()` 只负责"谁在什么时候干什么"。

**我怎么做：**

我的 `CodeReviewAgent.run()` 只有 3 行：

```python
def run(self, inputs):
    model_inputs = self.preprocess(inputs)     # scanner.scan()
    model_outputs = self._forward(model_inputs) # reviewer.review_files()
    return self.postprocess(model_outputs)      # reporter.generate()
```

编排是 Agent，实现是组件。泾渭分明。

---

## 模式 5：昂贵资源的共享单例

**Transformers 怎么做：**

模型只加载一次，所有 pipeline 共享。

**我怎么做：**

```python
_client: LLMClient | None = None

def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
```

一个 Anthropic 客户端，所有 Agent 共用。一次 .env 加载，一个连接池。

---

## 前后对比

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| 配置 | 散落在模块级变量里 | 类型安全的 dataclass，每个 Agent 一份 |
| API 客户端 | 每个文件自己创建 `Anthropic()` | 共享单例 `LLMClient` |
| Agent 结构 | 临时的函数串联 | `preprocess → _forward → postprocess` |
| 代码复用 | Agent 之间零复用 | 共享 `BaseAgent` + `LLMClient` |
| 新增 Agent | 基本重写 | 继承 `BaseAgent`，实现 3 个方法 |
| 代码行数 | - | 减少约 30%，但功能更多 |

---

## 真正的收获

读一个工程化良好的开源库源码，3 天的收获超过 3 个月的教程。

你不需要是资深工程师才能读源码。你只需要带着一个问题：**"他们是怎么组织这些代码的？为什么这样组织？"**

那些设计模式就藏在代码里，等着你去看。

---

*这是我的 [ai-agent-playground](https://github.com/aidless/ai-agent-playground) 系列的第 2 篇博客。从源码中学工程，在项目中练手。下一篇：RAG 知识库问答系统。*
