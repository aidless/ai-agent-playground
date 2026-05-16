"""LoRA Fine-Tuning — teach a large language model new skills efficiently.

LoRA = Low-Rank Adaptation. Instead of updating all 7 billion parameters
(which needs a datacenter GPU), LoRA only trains tiny "adapter" matrices
(~0.1% of parameters). Same results, 1/1000th the cost.
"""
