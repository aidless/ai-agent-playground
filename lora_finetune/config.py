"""LoRA fine-tuning configuration.

LoRA (Low-Rank Adaptation) explained for grandma:
  Imagine you have a huge cookbook (the base model, 7 billion parameters).
  You want to teach it Thai cooking. Instead of rewriting the entire cookbook,
  you add sticky notes (LoRA adapters) on relevant pages.
  The original text stays, but the sticky notes change the output.

  Sticky notes are tiny (~0.1% of the cookbook), so you can:
  - Train on a laptop instead of a datacenter
  - Save/load just the sticky notes (a few MB, not 14 GB)
  - Swap sticky notes to change cuisines (math → coding → poetry)
"""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """All the knobs and dials for fine-tuning."""

    # ---- Model ----
    # Which base model to fine-tune. Use a small one for CPU testing:
    #   "Qwen/Qwen2.5-0.5B"         (0.5B params, Chinese+English, ~1GB)
    #   "Qwen/Qwen2.5-1.5B"         (1.5B params, better quality)
    #   "google/gemma-2-2b-it"      (2B params, English-focused)
    # Or go big on Colab GPU:
    #   "Qwen/Qwen2.5-7B"           (7B params, needs T4 16GB + QLoRA)
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # ---- LoRA hyperparameters (the "sticky note size") ----
    # r (rank): how much new information each sticky note can hold
    #   r=8  → good for most tasks (recommended)
    #   r=16 → more capacity, slightly slower
    #   r=4  → very fast, for simple tasks only
    lora_r: int = 8

    # lora_alpha: scaling factor. Higher = stronger sticky notes
    #   Common rule: alpha = 2 × r
    lora_alpha: int = 16

    # dropout: randomly ignore some sticky notes during training (prevents overfitting)
    lora_dropout: float = 0.1

    # Which layers get sticky notes? "all" = every linear layer
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",       # FFN layers
    ])

    # ---- Training ----
    num_epochs: int = 3             # How many times to read the whole dataset
    batch_size: int = 4             # How many examples to process at once
    gradient_accumulation_steps: int = 4  # "Pretend bigger batch" (4×4=16 effective)
    learning_rate: float = 2e-4     # How fast to learn (too fast = overshoot, too slow = never converge)
    warmup_steps: int = 100         # Gradually increase learning rate at the start
    max_length: int = 512           # Max tokens per example (truncate longer)
    save_steps: int = 500           # Save checkpoint every N steps

    # ---- Memory saving ----
    use_4bit: bool = True           # QLoRA: compress model to 4-bit (saves 75% memory)
    use_gradient_checkpointing: bool = True  # Trade compute for memory

    # ---- Output ----
    output_dir: str = "./lora_output"
