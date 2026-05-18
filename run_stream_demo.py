import sys
import os
from ai_agent_playground.agents.chat_agent import SimpleChatAgent
from ai_agent_playground.config import BaseAgentConfig

config = BaseAgentConfig(model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"))
agent = SimpleChatAgent(config=config)

print("🤖 Agent: ", end="", flush=True)
try:
    # 消费生成器，实时打印
    for chunk in agent.run_stream("请用 150 字简述大语言模型的工作原理。"):
        print(chunk, end="", flush=True)
    print("\n✅ 流式输出完成")
except Exception as e:
    print(f"\n❌ 流式中断: {e}")