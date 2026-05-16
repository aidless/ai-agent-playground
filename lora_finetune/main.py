"""LoRA Fine-Tuning CLI.

Usage:
  uv run python -m lora_finetune.main train                    # Train with defaults
  uv run python -m lora_finetune.main chat ./lora_output       # Chat with model
  uv run python -m lora_finetune.main test                     # Quick test on CPU
"""

import sys
from .config import LoRAConfig


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "train":
        from .train import train
        config = LoRAConfig()
        # Parse extra args
        for arg in sys.argv[2:]:
            if arg.startswith("--model="):
                config.model_name = arg.split("=", 1)[1]
            elif arg.startswith("--epochs="):
                config.num_epochs = int(arg.split("=", 1)[1])
            elif arg.startswith("--output="):
                config.output_dir = arg.split("=", 1)[1]
            elif arg == "--no-4bit":
                config.use_4bit = False
        train(config)

    elif cmd == "chat":
        from .inference import chat
        model_path = sys.argv[2] if len(sys.argv) > 2 else "./lora_output"
        base_model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-0.5B"
        chat(model_path, base_model)

    elif cmd == "test":
        # Quick CPU test: train on a tiny model
        print("Running quick CPU test with tiny config...\n")
        from .train import train
        config = LoRAConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            use_4bit=False,
        )
        train(config)

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: train, chat, test")


if __name__ == "__main__":
    main()
