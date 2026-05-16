"""
Few-shot Example Manager — dynamically select and format examples.

Static few-shot prompts work OK. Dynamic selection works better:
  1. Embed the user's query
  2. Find the most similar examples in the pool
  3. Insert them into the prompt

This is essentially RAG for prompt examples — same concept, different target.

Usage:
    pool = FewShotPool("calculator")
    pool.add("Calculate 15*15", '{"tool": "calculator", "args": {"expression": "15*15"}}')
    pool.add("What is sqrt(144)?", '{"tool": "calculator", "args": {"expression": "sqrt(144)"}}')

    selector = FewShotSelector(pool)
    prompt = selector.build_prompt("What is 25*4?", num_examples=2)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    """One few-shot example."""

    input: str
    output: str
    meta: dict[str, Any] = field(default_factory=dict)


class FewShotPool:
    """Collection of examples for a specific task type."""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.examples: list[Example] = []

    def add(self, input_text: str, output_text: str, **meta):
        """Add an example to the pool."""
        self.examples.append(Example(
            input=input_text, output=output_text, meta=meta or {},
        ))

    def add_batch(self, pairs: list[tuple[str, str]]):
        """Add multiple (input, output) pairs at once."""
        for inp, out in pairs:
            self.add(inp, out)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class FewShotSelector:
    """Select the most relevant examples for a given query.

    Two selection strategies:
      - random:  pick randomly (fast, works as baseline)
      - semantic: embed query + examples, pick by cosine similarity (smart)
    """

    def __init__(self, pool: FewShotPool):
        self.pool = pool
        self._ef = None

    def _get_embedding_fn(self):
        if self._ef is None:
            import chromadb.utils.embedding_functions as ef
            self._ef = ef.DefaultEmbeddingFunction()
        return self._ef

    def select(
        self, query: str, num_examples: int = 2, strategy: str = "semantic"
    ) -> list[Example]:
        """Select the best examples for a query.

        Args:
            query: the user's input
            num_examples: how many examples to include
            strategy: "random" | "semantic"
        """
        if strategy == "random" or len(self.pool) <= num_examples:
            import random
            return random.sample(
                self.pool.examples,
                min(num_examples, len(self.pool)),
            )

        # Semantic selection: embed query + all examples, pick most similar
        return self._select_semantic(query, num_examples)

    def _select_semantic(
        self, query: str, num_examples: int
    ) -> list[Example]:
        """Select examples by embedding similarity."""
        ef = self._get_embedding_fn()

        inputs = [query] + [e.input for e in self.pool.examples]
        embeddings = ef(inputs)

        query_emb = embeddings[0]
        example_embs = embeddings[1:]

        scored = []
        for i, emb in enumerate(example_embs):
            sim = self._cosine_sim(query_emb, emb)
            scored.append((sim, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for _, idx in scored[:num_examples]:
            selected.append(self.pool.examples[idx])

        return selected

    def build_prompt(
        self,
        query: str,
        num_examples: int = 2,
        strategy: str = "semantic",
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Build a complete few-shot prompt.

        Args:
            query: the user's question
            num_examples: how many examples to include
            strategy: selection strategy
            prefix: text before the examples
            suffix: text after the examples (before the actual query)

        Returns a formatted prompt string.
        """
        examples = self.select(query, num_examples, strategy)

        parts = []
        if prefix:
            parts.append(prefix)
            parts.append("")

        parts.append("Examples:")
        parts.append("")

        for i, ex in enumerate(examples):
            parts.append(f"Q: {ex.input}")
            parts.append(f"A: {ex.output}")
            parts.append("")

        if suffix:
            parts.append(suffix)
            parts.append("")

        parts.append(f"Q: {query}")
        parts.append("A:")

        return "\n".join(parts)

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ============================================================
#  Pre-built example pools
# ============================================================


def create_calculator_pool() -> FewShotPool:
    """Example pool for calculator tool usage."""
    pool = FewShotPool("calculator")
    pool.add_batch([
        ("Calculate 15 * 15 + 12",
         '{"tool": "calculator", "args": {"expression": "15*15+12"}}'),
        ("What is the square root of 144?",
         '{"tool": "calculator", "args": {"expression": "sqrt(144)"}}'),
        ("Compute 3 to the power of 4",
         '{"tool": "calculator", "args": {"expression": "3^4"}}'),
        ("What is 100 divided by 7?",
         '{"tool": "calculator", "args": {"expression": "100/7"}}'),
        ("Calculate sin(30) + cos(60)",
         '{"tool": "calculator", "args": {"expression": "sin(30)+cos(60)"}}'),
        ("What is (25 * 4) + (100 / 5) - 10?",
         '{"tool": "calculator", "args": {"expression": "(25*4)+(100/5)-10"}}'),
    ])
    return pool


def create_file_ops_pool() -> FewShotPool:
    """Example pool for file operation tool usage."""
    pool = FewShotPool("file_ops")
    pool.add_batch([
        ("Read the file config.txt",
         '{"tool": "read_file", "args": {"path": "config.txt"}}'),
        ("Create a file called notes.txt with content 'Hello World'",
         '{"tool": "write_file", "args": {"path": "notes.txt", "content": "Hello World"}}'),
        ("What's in /etc/hosts?",
         '{"tool": "read_file", "args": {"path": "/etc/hosts"}}'),
        ("Save 'AI agents are powerful' to summary.md",
         '{"tool": "write_file", "args": {"path": "summary.md", "content": "AI agents are powerful"}}'),
    ])
    return pool


def create_general_pool() -> FewShotPool:
    """Mixed example pool for general agent tool selection."""
    pool = FewShotPool("general")
    pool.add_batch([
        ("Calculate 15*15",
         '{"tool": "calculator", "args": {"expression": "15*15"}}'),
        ("Read /etc/hosts",
         '{"tool": "read_file", "args": {"path": "/etc/hosts"}}'),
        ("Write 'hello' to greeting.txt",
         '{"tool": "write_file", "args": {"path": "greeting.txt", "content": "hello"}}'),
        ("Search for Python 3.13 features",
         '{"tool": "web_search", "args": {"query": "Python 3.13 features"}}'),
        ("What is AI?",
         "AI stands for Artificial Intelligence..."),
        ("How are you?",
         "I'm functioning well, thank you. How can I help?"),
    ])
    return pool
