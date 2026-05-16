"""
Local LLM — run models on-premise without sending data to external APIs.

When to use local models:
  - Data must stay within the corporate network (finance, healthcare, legal)
  - Need predictable latency (no API queue variability)
  - Cost-sensitive high-volume inference (no per-token billing)

Supported backends:
  - vLLM:           high-throughput serving with PagedAttention
  - llama.cpp:      CPU-friendly GGUF quantization
  - Ollama:         easy local setup, OpenAI-compatible API
  - HuggingFace:    direct Transformers inference (slow, for prototyping)

Quantization (sizes are approximate for a 7B model):
  - FP16:           ~14 GB VRAM (baseline)
  - GPTQ:           ~6 GB VRAM (INT4, good accuracy)
  - AWQ:            ~6 GB VRAM (INT4, better accuracy than GPTQ on some tasks)
  - GGUF Q4_K_M:    ~4 GB RAM (CPU, very accessible)
  - GGUF Q2_K:      ~3 GB RAM (CPU, worst quality, smallest size)

Usage:
    from ai_agent_playground.local_llm import LocalLLM, BackendType

    llm = LocalLLM(
        backend=BackendType.VLLM,
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="awq",
    )
    with llm:
        response = llm.chat([{"role": "user", "content": "Hello"}])
"""

import enum
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BackendType(str, enum.Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    HUGGINGFACE = "huggingface"


class Quantization(str, enum.Enum):
    NONE = "none"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF_Q4 = "q4_k_m"  # Q4_K_M — best quality/size ratio
    GGUF_Q5 = "q5_k_m"  # Q5_K_M — better quality, slightly larger
    GGUF_Q2 = "q2_k"  # Q2_K — smallest, lowest quality
    BNB_INT8 = "bitsandbytes_int8"  # 8-bit via bitsandbytes
    BNB_INT4 = "bitsandbytes_int4"  # 4-bit via bitsandbytes


@dataclass
class LocalLLMConfig:
    """Configuration for a local LLM deployment."""

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: BackendType = BackendType.VLLM
    quantization: Quantization = Quantization.NONE

    # vLLM settings
    vllm_tensor_parallel: int = 1  # Number of GPUs
    vllm_max_model_len: int = 8192
    vllm_gpu_memory_utilization: float = 0.90

    # llama.cpp settings
    llama_n_ctx: int = 8192
    llama_n_threads: int = 8
    llama_n_gpu_layers: int = 0  # 0 = CPU only, -1 = all layers on GPU

    # Ollama settings
    ollama_host: str = "http://localhost:11434"

    # Common
    temperature: float = 0.0
    max_tokens: int = 2048
    api_port: int = 8000


class LocalLLM:
    """Unified interface for local LLM backends.

    All backends expose an OpenAI-compatible chat completions API,
    so the core send/send_stream methods work identically regardless of backend.
    """

    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig()
        self._process: subprocess.Popen | None = None
        self._client = None  # OpenAI client, set after startup

    # ============================================================
    #  Lifecycle
    # ============================================================

    def start(self):
        """Start the local LLM server."""
        backend = self.config.backend
        if backend == BackendType.VLLM:
            self._start_vllm()
        elif backend == BackendType.OLLAMA:
            self._start_ollama()
        elif backend == BackendType.LLAMA_CPP:
            self._start_llama_cpp()
        elif backend == BackendType.HUGGINGFACE:
            self._start_huggingface()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def stop(self):
        """Stop the local LLM server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ============================================================
    #  Chat API (OpenAI-compatible)
    # ============================================================

    def chat(self, messages: list[dict]) -> str:
        """Send a chat completion request to the local model."""
        import json
        import urllib.request
        import urllib.error

        url = f"http://localhost:{self.config.api_port}/v1/chat/completions"
        payload = json.dumps({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to {self.config.backend.value} at {url}: {e}"
            )

    def is_healthy(self) -> bool:
        """Check if the local server is running."""
        import urllib.request
        import urllib.error

        try:
            url = f"http://localhost:{self.config.api_port}/health"
            with urllib.request.urlopen(url, timeout=3):
                return True
        except Exception:
            return False

    # ============================================================
    #  Backend launchers
    # ============================================================

    def _start_vllm(self):
        """Launch vLLM with PagedAttention for high-throughput serving."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model,
            "--tensor-parallel-size", str(self.config.vllm_tensor_parallel),
            "--max-model-len", str(self.config.vllm_max_model_len),
            "--gpu-memory-utilization", str(self.config.vllm_gpu_memory_utilization),
            "--port", str(self.config.api_port),
        ]
        if self.config.quantization != Quantization.NONE:
            cmd.extend(["--quantization", self._vllm_quant_flag()])

        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        print(f"Starting vLLM: {' '.join(cmd)}")

    def _start_ollama(self):
        """Pull and serve a model with Ollama."""
        import urllib.request
        import json

        # Pull model if not present
        pull_url = f"{self.config.ollama_host}/api/pull"
        pull_data = json.dumps({"name": self.config.model}).encode("utf-8")
        req = urllib.request.Request(pull_url, data=pull_data)
        try:
            with urllib.request.urlopen(req, timeout=300):
                pass
        except Exception as e:
            print(f"Ollama pull warning: {e}")

    def _start_llama_cpp(self):
        """Launch llama.cpp server (GGUF models)."""
        model_path = self._resolve_gguf_path()
        cmd = [
            "llama-server",
            "-m", model_path,
            "--ctx-size", str(self.config.llama_n_ctx),
            "--threads", str(self.config.llama_n_threads),
            "--n-gpu-layers", str(self.config.llama_n_gpu_layers),
            "--port", str(self.config.api_port),
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def _start_huggingface(self):
        """Launch HF Transformers via text-generation-inference (TGI)."""
        cmd = [
            "text-generation-launcher",
            "--model-id", self.config.model,
            "--port", str(self.config.api_port),
            "--max-total-tokens", str(self.config.vllm_max_model_len),
        ]
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    # ============================================================
    #  Helpers
    # ============================================================

    def _vllm_quant_flag(self) -> str:
        mapping = {
            Quantization.GPTQ: "gptq",
            Quantization.AWQ: "awq",
            Quantization.BNB_INT8: "bitsandbytes",
        }
        return mapping.get(self.config.quantization, "")

    def _resolve_gguf_path(self) -> str:
        """Resolve GGUF model path — checks HF cache and common locations."""
        # Check HF cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / f"models--{self.config.model.replace('/', '--')}"
        if model_dir.exists():
            gguf_files = list(model_dir.rglob("*.gguf"))
            if gguf_files:
                return str(gguf_files[0])
        return self.config.model  # Assume it's a path


# ============================================================
#  Quantization guide
# ============================================================


def quantization_guide() -> str:
    """Human-readable guide for choosing quantization."""
    return """
    Quantization Guide — trade quality for speed & smaller footprint
    ─────────────────────────────────────────────────────────────
    Format      Size (7B)   Quality    Speed     Best For
    ─────────────────────────────────────────────────────────────
    FP16         ~14 GB     ★★★★★     ★★★      Baseline, no compromise
    GPTQ-AWQ      ~6 GB     ★★★★      ★★★★     GPU inference, production
    GGUF Q5       ~5 GB     ★★★★      ★★★      CPU inference, good quality
    GGUF Q4       ~4 GB     ★★★½      ★★★★     CPU, best size/quality ratio
    GGUF Q2       ~3 GB     ★★        ★★★★★    CPU, minimum size
    BNB INT8      ~7 GB     ★★★★½     ★★★      Easy to setup, decent quality
    BNB INT4      ~4 GB     ★★★½      ★★★★     Easy to setup, small footprint
    ─────────────────────────────────────────────────────────────

    Decision tree:
    1. Have GPU with >=16GB VRAM? → vLLM + AWQ (best perf)
    2. Have GPU with 6-8GB VRAM?  → vLLM + GPTQ 4-bit
    3. CPU only, fast enough?      → llama.cpp + GGUF Q4_K_M
    4. CPU only, quality matters?  → llama.cpp + GGUF Q5_K_M
    5. Just prototyping?           → Ollama (automatic quant selection)
    """


# ============================================================
#  LoRA adapter manager
# ============================================================


class LoRAAdapter:
    """Manage LoRA adapters for domain-specific fine-tuning.

    LoRA (Low-Rank Adaptation) trains small adapter weights on top of
    a frozen base model. This means:
      - Training is fast (only ~1% of full fine-tuning parameters)
      - Storage is small (~10-200 MB per adapter)
      - You can swap adapters without reloading the base model

    Usage:
        adapter = LoRAAdapter(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            adapter_path="./lora/paint-advisor",
        )
        with adapter.serve():
            response = adapter.chat([...])
    """

    def __init__(self, base_model: str, adapter_path: str):
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)

    def serve(self, port: int = 8000):
        """Serve the base model with this LoRA adapter via vLLM."""
        import subprocess
        return subprocess.Popen([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.base_model,
            "--lora-modules", f"custom={self.adapter_path}",
            "--port", str(port),
        ])

    @staticmethod
    def train_config() -> dict:
        """Recommended LoRA training hyperparameters."""
        return {
            "r": 16,  # Rank — higher = more capacity, more parameters
            "lora_alpha": 32,  # Scaling factor
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "learning_rate": 2e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "epochs": 3,
        }
