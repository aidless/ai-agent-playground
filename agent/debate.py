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
        "Be thorough and precise. Show your reasoning step by step. "
        "After your proposal, the challenger will critique it."
    ),
    "challenger": (
        "You are a collaborative reviewer — NOT an adversary. Read the primary model's "
        "proposal and provide CONSTRUCTIVE step-by-step critique:\n"
        "1) Which reasoning steps are correct? (acknowledge strengths)\n"
        "2) Which steps have errors or gaps? (cite specific step)\n"
        "3) What edge cases are missing?\n"
        "4) What alternative approach could be better?\n"
        "Be rigorous but supportive — the goal is truth, not winning."
    ),
    "primary_rebuttal": (
        "The challenger has critiqued your proposal step by step. Address each point:\n"
        "1) Accept valid critiques and REFINE your reasoning\n"
        "2) Rebut mistaken critiques with evidence\n"
        "3) Integrate the challenger's good ideas into your improved proposal\n"
        "Produce your final proposal incorporating all valid feedback."
    ),
    "arbitrator": (
        "You are the final arbitrator. Review the full debate record and synthesize "
        "the best possible answer. Follow these rules:\n"
        "1) Adopt correct points from both sides\n"
        "2) When they agree, accept the consensus\n"
        "3) When they disagree, use independent reasoning\n"
        "4) Acknowledge uncertainty where it remains\n"
        "Output only the final answer — no meta-commentary."
    ),
}

# Process-centric debate (ColMAD/DynaDebate style) — focuses on step-by-step logic
COLMAD_PROMPTS = {
    "primary": (
        "Solve the task by showing your reasoning STEP BY STEP. "
        "Number each step. Be explicit about assumptions at each step. "
        "End with a clear conclusion."
    ),
    "challenger": (
        "Review the solution STEP BY STEP. For each step:\n"
        "- [CORRECT] if the logic is sound\n"
        "- [GAP] if something is missing\n"
        "- [ERROR] if the logic is wrong, explain why\n"
        "Also identify any MISSING steps that should have been included.\n"
        "Do NOT just vote on the final answer — critique the process."
    ),
    "synthesis": (
        "You have the original solution and a step-by-step critique. "
        "Synthesize an improved solution that:\n"
        "1. Keeps correct steps from the original\n"
        "2. Fixes steps marked [ERROR] or [GAP]\n"
        "3. Adds any MISSING steps identified\n"
        "4. Produces the final answer"
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

    async def debate_process_centric(self, task: str, primary_model: str, challenger_model: str) -> DebateResult:
        """Process-centric debate (ColMAD/DynaDebate style).

        Focuses on step-by-step logic critique instead of outcome voting.
        Collaborative, not adversarial — the goal is truth, not winning.
        """
        import uuid
        debate_id = f"coldmad-{uuid.uuid4().hex[:8]}"
        result = DebateResult(
            debate_id=debate_id, task=task,
            primary_model=primary_model, challenger_model=challenger_model,
        )
        start_time = time.time()

        try:
            # Round 1: Primary shows step-by-step reasoning
            step_by_step = await self._ask(
                self.primary_client, primary_model,
                COLMAD_PROMPTS["primary"], task,
            )
            result.rounds.append(DebateRound(1, "primary_steps", step_by_step, 0))

            # Round 2: Challenger critiques each step
            critique_context = f"TASK:\n{task}\n\nSOLUTION TO REVIEW (step by step):\n{step_by_step}"
            critique = await self._ask(
                self.challenger_client, challenger_model,
                COLMAD_PROMPTS["challenger"], critique_context,
            )
            result.rounds.append(DebateRound(2, "challenger_critique", critique, 0))

            # Round 3: Synthesis — incorporate corrections
            synth_context = (
                f"TASK:\n{task}\n\n"
                f"ORIGINAL SOLUTION:\n{step_by_step}\n\n"
                f"STEP-BY-STEP CRITIQUE:\n{critique}"
            )
            synthesis = await self._ask(
                self.primary_client, primary_model,
                COLMAD_PROMPTS["synthesis"], synth_context,
            )
            result.rounds.append(DebateRound(3, "synthesis", synthesis, 0))

            result.total_rounds = 3
            result.consensus = synthesis
            result.completed = True

        except Exception as e:
            result.error = str(e)
            logger.error("Process-centric debate %s failed: %s", debate_id, e)

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
