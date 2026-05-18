"""简单对话 Agent —— 支持同步 & 流式"""
from collections.abc import Generator
from ..base import BaseAgent, StreamItem
from ..config import BaseAgentConfig


class SimpleChatAgent(BaseAgent):
    def preprocess(self, inputs: str, **kwargs) -> dict:
        history = kwargs.get("history", [])
        return {"messages": history + [{"role": "user", "content": inputs}]}

    def _forward(self, model_inputs: dict, **kwargs) -> dict:
        """同步调用"""
        response = self.llm.chat.completions.create(
            model=self.config.model,
            messages=model_inputs["messages"],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024)
        )
        return response

    def _forward_stream(self, model_inputs: dict, **kwargs) -> Generator[StreamItem, None, None]:
        """🌊 流式调用核心实现"""
        response = self.llm.chat.completions.create(
            model=self.config.model,
            messages=model_inputs["messages"],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=True  # 🔑 开启 SSE 流式传输
        )

        for chunk in response:
            # 兼容部分厂商首块 delta.content 为 None 的情况
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content  # 逐 token 产出

    def postprocess(self, model_outputs, **kwargs) -> str:
        return model_outputs.choices[0].message.content