"""
HelloAgent —— 你写的第一个 AI Agent。

这是最简单的 Agent 示例，演示了"管道模式"是怎么工作的。

管道模式（Pipeline Pattern）：
  用户输入 → preprocess(准备) → _forward(调AI) → postprocess(格式化) → 回复

就像自动售货机：
  你投币 → 机器识别币值 → 出货 → 找零

HelloAgent 做的最简单的事：你问一个问题，AI 回答你。
"""

from typing import Any

from ai_agent_playground.base import BaseAgent  # Agent 骨架（三步走模板）

from .config import HelloAgentConfig  # HelloAgent 的配置盒


class HelloAgent(BaseAgent):
    """
    最简单的对话 Agent：问一句，答一句。

    继承 BaseAgent，只需要实现 3 个方法：
      preprocess  → 把用户文字包装成 API 请求格式
      _forward    → 发给 AI，拿回回复
      postprocess → 从回复里提取纯文本
    """

    # 告诉 BaseAgent："我用这个配置盒"
    config_class = HelloAgentConfig

    def __init__(self, config: HelloAgentConfig | None = None):
        # 调用父类的 __init__：加载配置、连接 AI 客户端
        super().__init__(config)

    # ============================================================
    #  三步 Pipeline 实现
    # ============================================================

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """
        第1步：把用户的文字包装成 API 需要的格式。

        用户说："你好" →
        包装成：{"messages": [{"role": "user", "content": "你好"}],
                 "model": "deepseek-v4-pro[1m]",
                 "max_tokens": 1024,
                 "system": "你是乐于助人的助手"}

        就像：去邮局寄信 → 你需要把信装进标准信封，写上地址。
        """
        return {
            "messages": [{"role": "user", "content": inputs}],
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        第2步：把准备好的数据发给 AI，拿回原始回复。

        **model_inputs 是 Python 的解包语法：
          model_inputs = {"model": "deepseek", "messages": [...]}
          self.llm.send(**model_inputs)
          ↓ 等价于 ↓
          self.llm.send(model="deepseek", messages=[...])

        就像：把信封投进邮筒。邮局帮你送，你等回信。
        """
        reply = self.llm.send(**model_inputs)
        # 返回字典：包含 AI 的原始回复和用户消息
        return {"reply": reply, "messages": model_inputs["messages"]}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """
        第3步：从 AI 的原始回复里提取纯文本。

        model_outputs = {"reply": "Hello! How can I help?", "messages": [...]}
        → 返回 "Hello! How can I help?"

        就像：收到回信 → 拆开信封 → 拿出信纸 → 读信。
        """
        return model_outputs["reply"]

    # ============================================================
    #  高级方法 —— 建立在 run() 之上
    #  这些方法不是必须的，但让 Agent 更好用
    # ============================================================

    def ask(self, question: str) -> str:
        """
        问一个问题，拿回答。单轮对话。

        相当于：打电话 → 问问题 → 听答案 → 挂电话。
        """
        return self.run(question)

    def chat(self):
        """
        多轮对话模式。记住之前的对话内容，持续聊下去。

        相当于：坐下来聊天，上下文都记得。
        输入 'quit' 退出，输入 'clear' 清空记忆。
        """
        print("=" * 60)
        print("  Hello Agent (Pipeline mode)")
        print("  Type 'quit' to exit, 'clear' to reset")
        print("=" * 60)
        print()

        history: list[dict] = []  # 对话历史，像聊天记录

        while True:  # 无限循环，直到用户说 quit
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):  # 用户按 Ctrl+C 或 Ctrl+D
                print("\nGoodbye!")
                break

            if not user_input:  # 空输入，跳过
                continue
            if user_input.lower() == "quit":  # 用户说"再见"
                print("Goodbye!")
                break
            if user_input.lower() == "clear":  # 用户说"忘了之前聊的"
                history = []
                print("[Conversation cleared]\n")
                continue

            # 把用户的消息加入历史
            history.append({"role": "user", "content": user_input})

            # 组装请求：这次要带全部历史，让 AI 知道上下文
            model_inputs = {
                "messages": list(history),  # list() 创建副本，防止意外修改
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "system": self.config.system_prompt,
            }
            # 调 AI，拿回复
            reply = self._forward(model_inputs)["reply"]
            print(f"AI: {reply}\n")

            # 把 AI 的回复也加入历史（下一轮 AI 能看到自己说过什么）
            history.append({"role": "assistant", "content": reply})


# ============================================================
#  如果你直接运行这个文件（python -m hello_agent.agent）
#  会执行下面的演示代码
# ============================================================
if __name__ == "__main__":
    agent = HelloAgent()  # 创建 Agent 实例
    print("Demo: Single question\n")
    answer = agent.ask("What is an AI agent? Answer in 2-3 sentences.")
    print(f"Q: What is an AI agent?\nA: {answer}\n")
    # 如果想进入聊天模式，取消下面这行的注释：
    # agent.chat()
