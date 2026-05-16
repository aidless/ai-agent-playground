"""AI Agent Playground — A framework for building AI agents with the Pipeline pattern.

Inspired by HuggingFace Transformers' design:
- Configs are typed dataclasses (like PreTrainedConfig)
- Agents follow preprocess → _forward → postprocess (like Pipeline)
- LLMClient centralizes API access (like PreTrainedModel)
"""
