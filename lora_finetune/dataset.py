"""Dataset preparation — turning raw text into training examples.

A "dataset" for LLM fine-tuning is just:
  - Input text (instruction + context)
  - Expected output text (what the model should say)

Like flashcards: question on front, answer on back.
"""

from torch.utils.data import Dataset


# ---- Built-in demo dataset: Chinese-English translation ----
# A tiny dataset to prove the pipeline works. Replace with your own data.
DEMO_DATA = [
    {"instruction": "Translate to English: 你好", "output": "Hello"},
    {"instruction": "Translate to English: 谢谢", "output": "Thank you"},
    {"instruction": "Translate to English: 再见", "output": "Goodbye"},
    {"instruction": "Translate to English: 苹果", "output": "Apple"},
    {"instruction": "Translate to English: 电脑", "output": "Computer"},
    {"instruction": "Translate to English: 今天天气真好", "output": "The weather is really nice today"},
    {"instruction": "Translate to English: 我喜欢学习AI", "output": "I like learning AI"},
    {"instruction": "Translate to English: 机器学习", "output": "Machine learning"},
    # Repeat to have enough data for training (20 examples minimum)
    {"instruction": "Translate to Chinese: Hello", "output": "你好"},
    {"instruction": "Translate to Chinese: Thank you", "output": "谢谢"},
    {"instruction": "Translate to Chinese: Goodbye", "output": "再见"},
    {"instruction": "Translate to Chinese: Computer", "output": "电脑"},
    {"instruction": "Translate to Chinese: The weather is nice", "output": "天气很好"},
    {"instruction": "Translate to Chinese: I love programming", "output": "我喜欢编程"},
    {"instruction": "Translate to Chinese: Artificial Intelligence", "output": "人工智能"},
    {"instruction": "Translate to Chinese: Deep learning is powerful", "output": "深度学习很强大"},
    {"instruction": "What is the capital of China?", "output": "Beijing"},
    {"instruction": "What is 2 + 2?", "output": "4"},
    {"instruction": "Name a fruit that is red.", "output": "Apple"},
    {"instruction": "What language is spoken in Japan?", "output": "Japanese"},
]


def create_prompt(instruction: str, output: str = "", add_generation_prompt: bool = False) -> str:
    """Format an instruction in Qwen chat format.

    The model expects a specific format:
      <|im_start|>system
      You are a helpful assistant.<|im_end|>
      <|im_start|>user
      {instruction}<|im_end|>
      <|im_start|>assistant
      {output}<|im_end|>
    """
    parts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        f"<|im_start|>user\n{instruction}<|im_end|>\n",
        "<|im_start|>assistant\n",
    ]
    if output:
        parts.append(f"{output}<|im_end|>\n")
    return "".join(parts)


class InstructionDataset(Dataset):
    """A PyTorch dataset that converts instructions into token IDs.

    Each item is a dict with:
      - input_ids: the full prompt (instruction + output), tokenized
      - labels: same as input_ids, but instruction part is masked (-100)
      - attention_mask: which tokens are real (vs padding)
    """

    def __init__(self, data: list[dict], tokenizer, max_length: int = 512):
        self.examples = []
        for item in data:
            # Format as chat prompt
            full_text = create_prompt(item["instruction"], item["output"])

            # Tokenize: text → numbers
            encoded = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]

            # Create labels: copy input_ids, then mask the instruction part
            # Mask = set to -100 (PyTorch ignores -100 in loss computation)
            labels = input_ids.clone()

            # Find where the assistant response starts ("<|im_start|>assistant\n")
            instruction_only = create_prompt(item["instruction"])
            inst_encoded = tokenizer(instruction_only, add_special_tokens=False)
            inst_len = len(inst_encoded["input_ids"])

            # Mask everything before and including the assistant marker
            labels[:inst_len] = -100

            # Also mask padding tokens
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
