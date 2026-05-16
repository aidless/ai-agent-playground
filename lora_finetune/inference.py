"""Run inference with a LoRA fine-tuned model.

After training, the fine-tuned model is just a small adapter file (a few MB).
Load it on top of the base model and start chatting.
"""

import sys
import torch
from .dataset import DEMO_DATA, create_prompt


def chat(model_path: str, base_model: str = "Qwen/Qwen2.5-0.5B"):
    """Interactive chat with a LoRA fine-tuned model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {base_model}")
    print(f"Loading LoRA adapter: {model_path}")
    print(f"Device: {device}")
    print()

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to(device)

    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    print("Model loaded. Type 'quit' to exit.\n")

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

        # Format prompt and generate
        prompt = create_prompt(user_input, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens (skip the prompt)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"AI: {response}\n")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./lora_output"
    base_model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B"
    chat(model_path, base_model)
