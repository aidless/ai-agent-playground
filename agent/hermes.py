"""Hermes 本地推理适配器（基于 HuggingFace Transformers）

用法:
    from agent.hermes import Hermes

    h = Hermes("NousResearch/Hermes-3-Llama-3.1-8B")
    print(h.generate("你好"))

支持的 Hermes 模型:
    - NousResearch/Hermes-3-Llama-3.1-8B
    - NousResearch/Hermes-3-Llama-3.1-8B-GGUF (需配合 llama.cpp)
    - teknium/OpenHermes-2.5-Mistral-7B
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Hermes:
    def __init__(self, model_name: str = "NousResearch/Hermes-3-Llama-3.1-8B", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self._pipe = None

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load(self):
        """懒加载 model + tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading {self.model_name} on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Loaded {self.model_name}")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        if self._pipe is None:
            self._load()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).removeprefix(prompt).strip()

    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        """OpenAI 风格的消息列表 → Hermes 回复"""
        # Hermes 3 原生支持 ChatML 格式
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return self.generate(prompt, max_new_tokens=max_new_tokens)
