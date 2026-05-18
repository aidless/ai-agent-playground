"""AttnRes Router — learned selective agent routing.

Inspired by Attention Residuals (Kimi/Moonshot AI, 2026).

Core insight: Standard multi-agent voting = uniform residual accumulation.
Each agent's output gets equal weight regardless of context. This is the
same problem AttnRes solves for transformer layers.

We replace uniform voting with learned softmax attention over agent outputs:
  final = Sigma alpha_i * agent_output_i
  alpha_i = softmax(query_task · key_i)

Where:
  - query: task-dependent routing vector (what does this task need?)
  - key: agent output summary (what did this agent produce?)
  - The attention weights tell us WHICH agent outputs are relevant NOW.

Block grouping mirrors Block AttnRes: agents in the same role-block share
intra-block communication, and only block-level summaries cross blocks.

Architecture:
    Task Context  -->  Pseudo-query q
                             |
    Agent₁ output --> key₁ --+--> alpha₁
    Agent₂ output --> key₂ --+--> alpha₂  --> softmax --> h = Sigma alpha_i * v_i
    Agent₃ output --> key₃ --+--> alpha₃

Comparison:
    Standard Voting:      h = (v1 + v2 + v3) / 3   (uniform, fixed)
    Weighted Voting:      h = w1*v1 + w2*v2 + w3*v3  (learned, static)
    AttnRes Routing:      h = alpha1*v1 + alpha2*v2 + alpha3*v3  (learned, context-dependent)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data types ──────────────────────────────────────

@dataclass
class AgentSignal:
    """Output from one agent, ready for routing."""
    agent_id: str
    role: str
    content: str
    confidence: float = 0.5
    summary_embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of selective routing over agent outputs."""
    signals: list[AgentSignal]
    weights: dict[str, float]         # agent_id -> attention weight
    aggregated: str                   # final aggregated text
    confidence: float                 # routing confidence
    routing_metadata: dict = field(default_factory=dict)


@dataclass
class BlockSummary:
    """Compressed summary of a role-block's outputs (like Block AttnRes)."""
    block_name: str
    role: str
    content: str                      # Aggregated block output
    agent_count: int
    signals: list[AgentSignal] = field(default_factory=list)


# ── Simple text-to-vector for routing ────────────────

def _text_vector(text: str, dim: int = 64) -> list[float]:
    """Deterministic text-to-vector for routing keys/queries.

    In production, this would use embeddings. For the routing mechanism,
    a lightweight hash-based projection is sufficient — the routing
    doesn't need semantic precision, just consistent differentiation.
    """
    h = hashlib.sha256(text.encode()).digest()
    vec = []
    for i in range(dim):
        byte_val = h[i % len(h)]
        # Use different hash positions for different dimensions
        pos = (i * 7 + 13) % len(h)
        vec.append((h[pos] / 255.0) * 2 - 1)  # [-1, 1]
    return vec


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


# ── Role-specific pseudo-queries ─────────────────────

ROLE_QUERIES = {
    # Each role has a learned "what do I look for in other agents?" vector
    # Derived from the role's semantic description via text_vector
    "planner": _text_vector("decompose complex goal into actionable structured steps dependencies risks execution order"),
    "developer": _text_vector("write working clean code implementation debug solve technical problems"),
    "reviewer": _text_vector("find bugs security issues anti-patterns correctness readability review"),
    "qa": _text_vector("test verify quality assurance edge cases regression coverage validation"),
    "devops": _text_vector("deploy monitor operate infrastructure reliability scaling automation"),
    "researcher": _text_vector("search find retrieve information facts evidence sources citations"),
    "proofreader": _text_vector("verify check factual accuracy correctness cross-reference validate claims evidence"),
}


# ── Role block definitions ──────────────────────────

ROLE_BLOCKS = {
    "planning": ["planner", "researcher"],
    "execution": ["developer", "devops"],
    "verification": ["reviewer", "qa", "proofreader"],
}


def get_block(role: str) -> str:
    for block_name, roles in ROLE_BLOCKS.items():
        if role in roles:
            return block_name
    return "general"


# ── AttnRes Router ──────────────────────────────────

class AttnResRouter:
    """Selective agent routing via learned attention over agent outputs.

    Usage:
        router = AttnResRouter()
        result = router.route(
            task="Build a REST API with input validation",
            signals=[agent1_output, agent2_output, agent3_output],
        )
        # result.weights shows which agents were most relevant
        # result.aggregated is the attention-weighted combination
    """

    def __init__(self, query_dim: int = 64, temperature: float = 0.5):
        self.query_dim = query_dim
        self.temperature = temperature
        self._route_history: list[RoutingResult] = []

    def route(
        self,
        task: str,
        signals: list[AgentSignal],
        mode: str = "full",        # full | block | residual (baseline)
    ) -> RoutingResult:
        """Route over agent signals using softmax attention.

        Args:
            task: The original task description
            signals: Agent outputs to route over
            mode: 'full' = dense attention, 'block' = inter-block only,
                  'residual' = uniform (baseline for comparison)
        """
        if not signals:
            return RoutingResult(
                signals=[], weights={}, aggregated="",
                confidence=0.0,
            )

        if mode == "residual":
            return self._residual_route(signals)
        elif mode == "block":
            return self._block_route(task, signals)
        else:
            return self._full_route(task, signals)

    def _full_route(self, task: str, signals: list[AgentSignal]) -> RoutingResult:
        """Full attention: every agent attends to every other agent (like Full AttnRes)."""
        # Build task query
        task_query = _text_vector(task, self.query_dim)

        # Compute attention weights
        scores = {}
        for sig in signals:
            # Key = agent role description + output content summary
            role_query = ROLE_QUERIES.get(sig.role, _text_vector(sig.role, self.query_dim))
            key = _text_vector(sig.content[:500] + sig.role, self.query_dim)

            # Score = task relevance * role specialization match
            task_sim = _cosine_sim(task_query, key)
            role_sim = _cosine_sim(task_query, role_query)

            # Combined score with temperature
            score = (0.7 * task_sim + 0.3 * role_sim) / self.temperature
            scores[sig.agent_id] = score

        # Softmax normalization
        weights = self._softmax(scores)

        # Weighted aggregation
        aggregated = self._aggregate(signals, weights)

        # Confidence = how sharp the attention distribution is
        confidence = self._compute_confidence(weights)

        result = RoutingResult(
            signals=signals,
            weights=weights,
            aggregated=aggregated,
            confidence=confidence,
            routing_metadata={"mode": "full", "temperature": self.temperature},
        )
        self._route_history.append(result)
        return result

    def _block_route(self, task: str, signals: list[AgentSignal]) -> RoutingResult:
        """Block routing: intra-block uniform, inter-block attention (like Block AttnRes).

        1. Group agent outputs into role-blocks
        2. Within each block, uniform aggregation → block summary
        3. Across blocks, softmax attention → final aggregation
        """
        # Step 1: Group into blocks
        blocks: dict[str, BlockSummary] = {}
        for sig in signals:
            block_name = get_block(sig.role)
            if block_name not in blocks:
                blocks[block_name] = BlockSummary(
                    block_name=block_name,
                    role=sig.role,
                    content="",
                    agent_count=0,
                )
            blocks[block_name].signals.append(sig)
            blocks[block_name].agent_count += 1

        # Step 2: Intra-block uniform aggregation
        for bname, bsum in blocks.items():
            if bsum.signals:
                bsum.content = self._residual_route(bsum.signals).aggregated

        # Step 3: Inter-block attention
        task_query = _text_vector(task, self.query_dim)
        block_scores = {}
        for bname, bsum in blocks.items():
            key = _text_vector(bsum.content[:500] + bname, self.query_dim)
            block_scores[bname] = _cosine_sim(task_query, key) / self.temperature

        block_weights = self._softmax(block_scores)

        # Step 4: Final aggregation — each signal's weight = block_weight / signals_in_block
        final_weights = {}
        for bname, bw in block_weights.items():
            bsum = blocks[bname]
            per_signal = bw / max(bsum.agent_count, 1)
            for sig in bsum.signals:
                final_weights[sig.agent_id] = per_signal

        aggregated = self._aggregate(signals, final_weights)
        confidence = self._compute_confidence(block_weights)

        result = RoutingResult(
            signals=signals,
            weights=final_weights,
            aggregated=aggregated,
            confidence=confidence,
            routing_metadata={
                "mode": "block",
                "blocks": list(blocks.keys()),
                "block_weights": block_weights,
            },
        )
        self._route_history.append(result)
        return result

    def _residual_route(self, signals: list[AgentSignal]) -> RoutingResult:
        """Uniform residual: all signals equal weight (baseline, like standard residuals)."""
        n = len(signals)
        weights = {sig.agent_id: 1.0 / n for sig in signals}
        aggregated = self._aggregate(signals, weights)

        return RoutingResult(
            signals=signals,
            weights=weights,
            aggregated=aggregated,
            confidence=1.0 / n,  # Low confidence = flat distribution
            routing_metadata={"mode": "residual"},
        )

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        """Compute softmax over scores."""
        if not scores:
            return {}
        values = list(scores.values())
        max_val = max(values)
        exp_sum = sum(2.71828 ** ((v - max_val)) for v in values)  # exp(x - max)
        if exp_sum == 0:
            n = len(scores)
            return {k: 1.0 / n for k in scores}
        return {k: (2.71828 ** ((scores[k] - max_val))) / exp_sum for k in scores}

    def _aggregate(self, signals: list[AgentSignal], weights: dict[str, float]) -> str:
        """Weighted aggregation of agent outputs."""
        if not signals:
            return ""

        parts = []
        for sig in signals:
            w = weights.get(sig.agent_id, 0.0)
            if w > 0.01 and sig.content.strip():
                parts.append(f"[{sig.role} weight={w:.2f}]\n{sig.content}")

        if not parts:
            return signals[0].content if signals else ""

        return "\n\n".join(parts)

    def _compute_confidence(self, weights: dict[str, float]) -> float:
        """Confidence = max_weight / avg_weight. High = sharp distribution."""
        if not weights:
            return 0.0
        values = list(weights.values())
        avg = sum(values) / len(values)
        if avg == 0:
            return 0.0
        return min(1.0, max(values) / avg)

    def compare(self, task: str, signals: list[AgentSignal]) -> dict:
        """Compare all three routing modes."""
        results = {}
        for mode in ["residual", "block", "full"]:
            r = self.route(task, signals, mode=mode)
            results[mode] = {
                "weights": {sig.agent_id: round(r.weights.get(sig.agent_id, 0), 3)
                           for sig in signals},
                "confidence": round(r.confidence, 3),
                "top_agent": max(r.weights, key=r.weights.get) if r.weights else None,
            }
        return results

    def history(self, limit: int = 20) -> list[RoutingResult]:
        return self._route_history[-limit:]
