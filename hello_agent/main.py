"""
Hello Agent — Your first AI agent that can think and respond.

This demonstrates the core pattern: define a system prompt → send a
user message → get the model's response. Think of it as the "Hello
World" of AI agents.
"""

import os
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

client = Anthropic(
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

SYSTEM_PROMPT = """\
You are a helpful AI assistant. When you answer:
- Be concise but thorough
- Use examples when they help explain a concept
- If you don't know something, say so honestly
"""


def _get_text(response) -> str:
    """Extract the text reply from a response that may contain thinking blocks."""
    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text
    return "[No text in response]"


def ask(message: str, model: str = "deepseek-v4-pro[1m]") -> str:
    """Send a message to the AI and get the response."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )
    return _get_text(response)


def chat():
    """Simple interactive chat loop. Type 'quit' to exit."""
    print("=" * 60)
    print("  AI Agent Playground — Hello Agent")
    print("  Type 'quit' to exit, 'clear' to reset")
    print("=" * 60)
    print()

    messages = []

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
            messages = []
            print("[Conversation cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        response = client.messages.create(
            model="deepseek-v4-pro[1m]",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        reply = _get_text(response)
        print(reply)
        print()

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    # Quick demo: single question
    print("Demo: Single question\n")
    answer = ask("What is an AI agent? Answer in 2-3 sentences.")
    print(f"Q: What is an AI agent?\nA: {answer}\n")

    # Uncomment the next line for interactive mode:
    # chat()
