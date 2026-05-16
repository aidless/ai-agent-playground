"""LoRA fine-tuning script — the core training loop.

This is the engine room. It:
  1. Loads a pre-trained model (e.g., Qwen2.5-0.5B)
  2. Attaches LoRA adapters (tiny trainable "sticky notes")
  3. Trains on your dataset
  4. Saves the adapters (just a few MB!)

Can run on:
  - CPU: very slow but works for tiny models + tiny datasets
  - Google Colab (free T4 GPU): 1-2 hours for a 0.5B model
  - Your own GPU: minutes

Google Colab quick start (copy-paste into a Colab cell):
  !git clone https://github.com/aidless/ai-agent-playground.git
  %cd ai-agent-playground
  !uv pip install peft accelerate transformers torch datasets
  !python -m lora_finetune.train --model Qwen/Qwen2.5-0.5B
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

from .config import LoRAConfig
from .dataset import DEMO_DATA, InstructionDataset


def train(config: LoRAConfig | None = None):
    """Run the full LoRA fine-tuning pipeline."""
    config = config or LoRAConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  LoRA Fine-Tuning Pipeline")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Device: {device}")
    print(f"  4-bit: {config.use_4bit}")
    print("=" * 70)
    print()

    # ---- Step 1: Load tokenizer ----
    print("[1/5] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    # Qwen doesn't have a default pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print()

    # ---- Step 2: Load model with quantization ----
    print("[2/5] Loading model...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if config.use_4bit and device == "cuda":
        # QLoRA: 4-bit quantization → fits 7B model on 16GB GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quant = even smaller
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("  Loaded with 4-bit quantization (QLoRA mode)")
    else:
        # Full precision (works on CPU, but slow and memory-heavy)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            model = model.to(device)
        print(f"  Loaded in {'float32 (CPU)' if device == 'cpu' else 'float16 (GPU)'}")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ---- Step 3: Attach LoRA adapters ----
    print("[3/5] Attaching LoRA adapters...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")
    print(f"  That's like adding {trainable/1e6:.1f}M sticky notes to a {total/1e9:.1f}B model")
    print()

    # Enable gradient checkpointing (saves memory, costs ~20% more compute)
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")
        print()

    # ---- Step 4: Prepare dataset ----
    print("[4/5] Preparing dataset...")
    train_dataset = InstructionDataset(DEMO_DATA, tokenizer, config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Batches: {len(train_loader)} (batch_size={config.batch_size})")
    print()

    # ---- Step 5: Train! ----
    print("[5/5] Training...")
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(config.warmup_steps, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = outputs.loss

            # Gradient accumulation: scale loss so effective batch = batch_size × grad_accum
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * config.gradient_accumulation_steps

            # Progress
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item()*config.gradient_accumulation_steps:.4f} "
                      f"avg={avg:.4f}")

            # Save checkpoint
            if global_step > 0 and global_step % config.save_steps == 0:
                model.save_pretrained(f"{config.output_dir}/checkpoint-{global_step}")
                print(f"  [Saved checkpoint at step {global_step}]")

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch} complete — avg loss: {avg_loss:.4f}")
        print()

    # ---- Save final model ----
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"\nModel saved to {config.output_dir}/")
    print(f"Files: adapter_config.json, adapter_model.safetensors (~{trainable/1e6:.1f} MB)")
    print()
    print("To use the fine-tuned model:")
    print(f"  uv run python -m lora_finetune.inference --model {config.output_dir}")
    print()
    print("Done!")

    return model, tokenizer


if __name__ == "__main__":
    config = LoRAConfig()

    # Parse CLI args
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            config.model_name = arg.split("=", 1)[1]
        elif arg.startswith("--epochs="):
            config.num_epochs = int(arg.split("=", 1)[1])
        elif arg.startswith("--output="):
            config.output_dir = arg.split("=", 1)[1]
        elif arg == "--no-4bit":
            config.use_4bit = False

    train(config)
