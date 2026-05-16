"""
MCP Tool-Use Agent —— 能使用工具的 AI Agent。

普通的 AI 只能"说"，这个 AI 能"做"：
  搜索网页、读写文件、执行命令、算数学题...

核心机制：ReAct 循环（Reasoning + Acting = 推理 + 行动）

  想象你让助手"帮我查一下今天天气，然后写进文件"：
    1. 助手想："我需要先搜索天气"（Reasoning）
    2. 助手打开搜索引擎查天气（Acting）
    3. 助手看到结果："哦，25°C 晴天"
    4. 助手写进 weather.txt（Acting）
    5. 助手回复："已保存到 weather.txt"（Final Answer）

  这就是 ReAct 循环——AI 在"思考"和"行动"之间来回切换，直到任务完成。

这个模式 Claude Code、ChatGPT、Copilot 都在用。
"""

import json    # 解析 AI 返回的 JSON（工具调用请求）
import re      # 正则表达式（从 AI 回复里提取 JSON）
from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import MCPAgentConfig
from .tools import TOOLS, TOOL_DESCRIPTIONS  # 工具注册表 + 工具使用说明


class MCPToolAgent(BaseAgent):
    """
    能使用工具的 Agent。

    可用的工具：
      web_search("Python教程") → 搜索网页
      read_file("notes.txt")      → 读文件
      write_file("a.txt", "hi")  → 写文件
      run_command("dir")          → 执行命令
      calculator("2+3*4")         → 算数学题

    工作流程（ReAct 循环）：
      preprocess:  把用户问题包好
      _forward:    ReAct 循环（最长 max_tool_rounds 轮）
      postprocess: 格式化答案 + 列出用了哪些工具
    """

    config_class = MCPAgentConfig

    def __init__(self, config: MCPAgentConfig | None = None):
        super().__init__(config)
        self.tools = TOOLS  # 工具字典：{"tool_name": function, ...}

    # ============================================================
    #  三步 Pipeline
    # ============================================================

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """把用户问题放进字典，准备处理。"""
        return {"question": inputs}

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        ReAct 循环：思考 → 行动 → 观察 → 再思考 ... → 最终回答。

        这是整个项目最核心的"智能"体现：
          AI 不只是你问一句它答一句，而是能自主决定"我需要什么信息"、
          主动调用工具获取信息、根据结果再调整策略。

        就像一个聪明的实习生：
          - 先分析问题："这个任务需要什么？"
          - 如果需要查资料，就去查
          - 查完看看结果够不够，不够再查
          - 最后汇总，给你一个完整的答案
        """
        question = model_inputs["question"]

        # ---- 构造第一轮对话：告诉 AI 它有哪些工具可用 ----
        # 这就像给实习生一份"可用资源清单"：
        # "你可以用：搜索引擎、文件系统、计算器..."
        conversation = [
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Available tools:\n" +
                "\n".join(f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()) +
                f"\n\nTo use a tool, respond with JSON:\n"
                f'{{"tool": "tool_name", "args": {{"arg": "value"}}}}\n'
                f"After getting tool results, give your final answer."
            )}
        ]

        tool_use_log = []  # 记录每一步用了什么工具（用于汇报）

        # ---- ReAct 循环：最多 max_tool_rounds 轮 ----
        for round_num in range(self.config.max_tool_rounds):
            # ① 问 AI：你要直接回答，还是先调工具？
            reply = self.llm.send(
                messages=conversation,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
            )

            # ② 检查 AI 是否要调用工具
            tool_call = self._parse_tool_call(reply)

            if tool_call is None:
                # AI 没有要调工具 → 这就是最终答案！
                return {
                    "answer": reply,
                    "tool_rounds": round_num,
                    "tool_log": tool_use_log,
                }

            # ③ AI 要调工具 → 执行工具
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]

            # 检查工具是否存在（AI 有时会编造不存在的工具名）
            if tool_name not in self.tools:
                result = f"Unknown tool: {tool_name}. Available: {list(self.tools)}"
            else:
                try:
                    # 真正调用工具！比如真的去搜索网页、读文件
                    result = self.tools[tool_name](**tool_args)
                except Exception as e:
                    # 工具执行失败（比如文件不存在、网络超时）
                    result = f"Tool error: {e}"

            # ④ 记录本次工具调用
            tool_use_log.append({
                "round": round_num + 1,
                "tool": tool_name,
                "args": tool_args,
                "result": str(result)[:500],  # 截断太长的结果
            })

            # ⑤ 把工具执行结果发回给 AI，让它继续思考
            conversation.append({"role": "assistant", "content": reply})
            conversation.append({
                "role": "user",
                "content": f"Tool result from {tool_name}:\n{result}\n\nBased on this, provide your answer."
            })
            # 回到循环开头 ①，AI 会决定：继续调工具还是给出最终答案

        # 达到最大轮数还没停 → 强制要求 AI 给出最终答案
        final = self.llm.send(
            messages=conversation + [
                {"role": "user", "content": "Please give your final answer now."}
            ],
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
        )
        return {
            "answer": final,
            "tool_rounds": self.config.max_tool_rounds,
            "tool_log": tool_use_log,
        }

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """
        格式化最终输出：答案 + 工具使用摘要。

        输出示例：
          答案是 237。

          ---
          *Tool calls made:*
          - calculator(expression=15*15+12)
            → 237...
        """
        answer = model_outputs["answer"]
        log = model_outputs["tool_log"]

        if not log:  # 没用到工具，直接返回答案
            return answer

        # 列出用了哪些工具（方便你理解 AI 是怎么思考的）
        lines = [answer, "", "---", "*Tool calls made:*"]
        for entry in log:
            args_str = ", ".join(f"{k}={v}" for k, v in entry["args"].items())
            lines.append(f"- `{entry['tool']}({args_str})`")
            preview = entry["result"][:150].replace("\n", " ")
            lines.append(f"  → {preview}...")
        return "\n".join(lines)

    # ============================================================
    #  工具调用解析器 —— 从 AI 回复里提取 JSON
    #
    #  AI 有时候返回纯 JSON：{"tool": "calculator", ...}
    #  有时候返回 markdown 代码块：```json\n{"tool": "calculator", ...}\n```
    #  我们需要两种情况都能处理
    # ============================================================

    @staticmethod
    def _parse_tool_call(text: str) -> dict | None:
        """
        从 AI 的文本回复里提取工具调用 JSON。

        像从一段话里找到"执行指令"——AI 可以说很多，但真正的指令在 JSON 里。

        支持两种格式：
          1. 裸 JSON：{"tool": "web_search", "args": {"query": "Python"}}
          2. Markdown 代码块：```json\n{"tool": "web_search", ...}\n```

        返回: {"tool": "xxx", "args": {...}} 或 None（没找到工具调用）
        """
        # 先找 markdown 代码块（AI 最喜欢这种格式）
        # re.DOTALL 让 . 也能匹配换行符
        block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if block_match:
            try:
                return json.loads(block_match.group(1))  # 解析 JSON
            except json.JSONDecodeError:
                pass  # 解析失败 → 继续尝试下一种

        # 再找行内 JSON（匹配 {"tool": "xxx", "args": {...}} 模式）
        for match in re.finditer(
            r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}\}',
            text
        ):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

        return None  # 没找到任何工具调用

    # ============================================================
    #  高级方法
    # ============================================================

    def ask(self, question: str) -> str:
        """问一个问题——Agent 可能会自动使用工具来回答。"""
        return self.run(question)
