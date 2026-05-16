"""HelloAgent — first contact with the AI, Pipeline-style.

Demonstrates the pattern: preprocess → _forward → postprocess.
"""

from typing import Any

from ai_agent_playground.base import BaseAgent

from .config import HelloAgentConfig


class HelloAgent(BaseAgent):
    """Simple conversational agent: ask a question, get an answer."""

    config_class = HelloAgentConfig

    def __init__(self, config: HelloAgentConfig | None = None):
        super().__init__(config)

    # ---- Pipeline implementation ----

    def preprocess(self, inputs: str, **kwargs) -> dict[str, Any]:
        """String message → API-ready format."""
        return {
            "messages": [{"role": "user", "content": inputs}],
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
        }

    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Call the LLM."""
        reply = self.llm.send(**model_inputs)
        return {"reply": reply, "messages": model_inputs["messages"]}

    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> str:
        """Extract the text reply."""
        return model_outputs["reply"]

    # ---- Higher-level methods (built on top of run) ----

    def ask(self, question: str) -> str:
        """Single-turn: ask a question, get the answer."""
        return self.run(question)

    def chat(self):
        """Interactive multi-turn chat loop."""
        print("=" * 60)
        print("  Hello Agent (Pipeline mode)")
        print("  Type 'quit' to exit, 'clear' to reset")
        print("=" * 60)
        print()

        history: list[dict] = []

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                history = []
                print("[Conversation cleared]\n")
                continue

            history.append({"role": "user", "content": user_input})

            model_inputs = {
                "messages": list(history),
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "system": self.config.system_prompt,
            }
            reply = self._forward(model_inputs)["reply"]
            print(f"AI: {reply}\n")

            history.append({"role": "assistant", "content": reply})


# ---- Script entry point ----
if __name__ == "__main__":
    agent = HelloAgent()
    print("Demo: Single question\n")
    answer = agent.ask("What is an AI agent? Answer in 2-3 sentences.")
    print(f"Q: What is an AI agent?\nA: {answer}\n")
    # agent.chat()
