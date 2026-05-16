"""Centralized LLM client — single Anthropic client shared across all agents.

Like how transformers shares PreTrainedModel across pipelines.
"""

import os
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

# Load once at module level
_load_dotenv_done = False


def _ensure_dotenv():
    global _load_dotenv_done
    if not _load_dotenv_done:
        # Walk up to find .env
        load_dotenv(Path(__file__).parent.parent / ".env")
        _load_dotenv_done = True


class LLMClient:
    """Thin wrapper around Anthropic client. Handles auth, base URL, and
    extracting text from responses that may contain thinking blocks."""

    def __init__(self):
        _ensure_dotenv()
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError(
                "DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY must be set in .env file. "
                "Copy .env.example to .env and fill in your keys."
            )
        self._client = Anthropic(base_url=base_url, api_key=api_key)

    def send(
        self,
        messages: list[dict],
        *,
        model: str = "deepseek-v4-pro[1m]",
        max_tokens: int = 2048,
        system: str = "",
    ) -> str:
        """Send messages to the model and return the text response."""
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Get text from response, collecting all text blocks."""
        parts = []
        for block in response.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
        return "\n".join(parts) if parts else "[No text in response]"


# Module-level singleton (like a shared model in transformers)
_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Get or create the shared LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
