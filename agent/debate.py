"""Multi-Model Debate — two models debate, third arbitrates.

Unlike cross-reviewer (which checks generated text against source),
debate combines multiple models to solve the same problem:

  1. Primary model proposes solution
  2. Challenger model critiques the proposal
  3. Primary model responds to critique
  4. Arbitrator model synthesizes final consensus

This is inspired by Anthropic's constitutional AI debate and
Meta's multi-agent self-play research. Different models catch
each other's blind spots.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DebateRound:
    round_num: int
    speaker: str         # "primary" | "challenger" | "arbitrator"
    content: str
    latency_ms: float


@dataclass
class DebateResult:
    debate_id: str
    task: str
    rounds: list[DebateRound] = field(default_factory=list)
    consensus: str = ""
    primary_model: str = ""
    challenger_model: str = ""
    arbitrator_model: str = ""
    total_rounds: int = 0
    total_latency_ms: float = 0.0
    completed: bool = False
    error: str = ""


DEBATE_SYSTEM_PROMPTS = {
    "primary": (
        "You are a skilled AI assistant. Propose a solution to the user's task. "
        "Be thorough and precise. After your proposal, the challenger will critique it."
    ),
    "challenger": (
        "You are a critical reviewer. Read the primary model's proposal and identify: "
        "1) factual errors, 2) missing edge cases, 3) better alternative approaches, "
        "4) unsafe or incorrect recommendations. Be constructive but rigorous."
    ),
    "primary_rebuttal": (
        "The challenger has critiqued your proposal. Address each point: "
        "accept valid critiques and refine your answer, rebut mistaken critiques. "
        "Produce your improved final proposal."
    ),
    "arbitrator": (
        "You are the final arbitrator. Review the debate and produce the best possible answer. "
        "Adopt correct points from both sides. When they disagree, use your own judgment. "
        "Output only the final answer — no meta-commentary about the debate process."
    ),
}


class DebateEngine:
    """Runs multi-model debates and returns consensus results.

    Usage:
        engine = DebateEngine(primary_client, challenger_client, arbitrator_client)
        result = await engine.debate(
            task="Write a function to sort a list",
            primary_model="deepseek-chat",
            challenger_model="qwen2.5:7b",
            arbitrator_model="deepseek-chat",
        )
    """

    def __init__(
        self,
        primary_client,
        challenger_client,
        arbitrator_client=None,
        max_rounds: int = 3,
    ):
        self.primary_client = primary_client
        self.challenger_client = challenger_client
        self.arbitrator_client = arbitrator_client or primary_client
        self.max_rounds = max_rounds
        self._results: dict[str, DebateResult] = {}

    async def debate(
        self,
        task: str,
        primary_model: str,
        challenger_model: str,
        arbitrator_model: str = "",
        context: str = "",
    ) -> DebateResult:
        """Run a full debate and return the consensus result."""
        import uuid
        debate_id = f"debate-{uuid.uuid4().hex[:8]}"
        result = DebateResult(
            debate_id=debate_id,
            task=task,
            primary_model=primary_model,
            challenger_model=challenger_model,
            arbitrator_model=arbitrator_model or primary_model,
        )
        start_time = time.time()

        try:
            # Round 1: Primary proposes
            primary_proposal = await self._ask(
                self.primary_client, primary_model,
                DEBATE_SYSTEM_PROMPTS["primary"],
                task, context,
            )
            result.rounds.append(DebateRound(1, "primary", primary_proposal, 0))
            result.total_rounds = 1

            # Round 2: Challenger critiques
            critique_context = f"TASK:\n{task}\n\nPRIMARY PROPOSAL:\n{primary_proposal}"
            critique = await self._ask(
                self.challenger_client, challenger_model,
                DEBATE_SYSTEM_PROMPTS["challenger"],
                critique_context,
            )
            result.rounds.append(DebateRound(2, "challenger", critique, 0))
            result.total_rounds = 2

            # Round 3: Primary rebuttal
            rebuttal_context = (
                f"TASK:\n{task}\n\n"
                f"YOUR ORIGINAL PROPOSAL:\n{primary_proposal}\n\n"
                f"CHALLENGER CRITIQUE:\n{critique}"
            )
            rebuttal = await self._ask(
                self.primary_client, primary_model,
                DEBATE_SYSTEM_PROMPTS["primary_rebuttal"],
                rebuttal_context,
            )
            result.rounds.append(DebateRound(3, "primary_rebuttal", rebuttal, 0))
            result.total_rounds = 3

            # Round 4: Arbitrator synthesizes
            arb_context = (
                f"TASK:\n{task}\n\n"
                f"PRIMARY PROPOSAL:\n{primary_proposal}\n\n"
                f"CHALLENGER CRITIQUE:\n{critique}\n\n"
                f"PRIMARY REBUTTAL:\n{rebuttal}"
            )
            consensus = await self._ask(
                self.arbitrator_client, result.arbitrator_model,
                DEBATE_SYSTEM_PROMPTS["arbitrator"],
                arb_context,
                task,
            )
            result.rounds.append(DebateRound(4, "arbitrator", consensus, 0))
            result.total_rounds = 4
            result.consensus = consensus
            result.completed = True

        except Exception as e:
            result.error = str(e)
            logger.error("Debate %s failed: %s", debate_id, e)

        result.total_latency_ms = (time.time() - start_time) * 1000
        self._results[debate_id] = result
        return result

    async def _ask(self, client, model: str, system_prompt: str, user_content: str, extra_context: str = "") -> str:
        """Send a prompt to a model and return the text response."""
        messages = [{"role": "system", "content": system_prompt}]
        if extra_context:
            messages.append({"role": "user", "content": extra_context})
        messages.append({"role": "user", "content": user_content})

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Debate model call failed (%s/%s): %s", model, system_prompt[:20], e)
            raise

    def get_result(self, debate_id: str) -> Optional[DebateResult]:
        return self._results.get(debate_id)

    def status(self) -> dict:
        return {
            "completed_debates": len(self._results),
            "recent": [
                {"id": r.debate_id, "rounds": r.total_rounds, "completed": r.completed}
                for r in list(self._results.values())[-5:]
            ],
        }
