"""Hermes 本地推理适配器（支持 HuggingFace Transformers 和 Ollama）

用法:
    from agent.hermes import Hermes

    # HuggingFace 后端
    h = Hermes("NousResearch/Hermes-3-Llama-3.1-8B")
    print(h.chat([{"role": "user", "content": "你好"}]))

    # Ollama 后端（需装 Ollama + 已 pull 模型）
    h = Hermes("qwen2.5:7b", backend="ollama")
    print(h.chat([{"role": "user", "content": "你好"}]))

支持的模型:
    - NousResearch/Hermes-3-Llama-3.1-8B（HuggingFace）— 需要 ~16GB 内存
    - hermes3（Ollama）— 需要 ~16GB 内存（经测试你机器 7.1GB 不够）
    - qwen2.5:7b（Ollama，推荐）— Q4 量化仅 ~4-5GB，中文出色 ✅
    - qwen2.5:3b / llama3.2:3b（Ollama）— 更低配置友好
    - teknium/OpenHermes-2.5-Mistral-7B（HuggingFace）
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class Hermes:
    def __init__(
        self,
        model_name: str = "NousResearch/Hermes-3-Llama-3.1-8B",
        backend: str = "huggingface",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self._loaded = False

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── HuggingFace 后端 ──

    def _load_hf(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # HF 镜像兜底：国内用户优先用 hf-mirror
        endpoint = os.environ.get("HF_ENDPOINT") or (
            "https://hf-mirror.com"
            if os.environ.get("HF_MIRROR", "1") == "1"
            else "https://huggingface.co"
        )

        logger.info("Loading %s via %s on %s...", self.model_name, endpoint, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        ).to(self.device)
        logger.info("Loaded %s", self.model_name)

    def _generate_hf(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        full = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去掉 prompt 部分，只保留新生成的内容
        prompt_len = len(self._tokenizer(prompt)["input_ids"])
        return self._tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    # ── Ollama 后端 ──

    def _generate_ollama(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        import httpx

        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "options": {"num_predict": max_new_tokens, "temperature": temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        return "".join(json.loads(line)["response"] for line in lines if line.strip())

    def _chat_ollama(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        import httpx

        resp = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "options": {"num_predict": max_new_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        return "".join(json.loads(line)["message"]["content"] for line in lines if line.strip())

    # ── 统一接口 ──

    def _ensure_loaded(self):
        if not self._loaded:
            if self.backend == "huggingface":
                self._load_hf()
            self._loaded = True

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        self._ensure_loaded()

        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_new_tokens, temperature)
        return self._generate_hf(prompt, max_new_tokens, temperature)

    def chat(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        if self.backend == "ollama":
            return self._chat_ollama(messages, max_new_tokens)

        # HuggingFace: 用 tokenizer 的 chat_template（比手拼 ChatML 可靠）
        self._ensure_loaded()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._generate_hf(prompt, max_new_tokens=max_new_tokens)
