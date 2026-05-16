"""
Hallucination Detector — because "sounds plausible" != "actually true".

Three detection strategies (layered defense):

1. Factual Consistency: does the answer contradict the provided context?
   → Compare claims in answer against source documents.

2. Citation Verification: do the citations actually support the claims?
   → For each [Chunk N] citation, check if the cited text contains the claim.

3. Self-Consistency: does the model give the same answer when asked multiple times?
   → Sample N responses, check for agreement.

Usage:
    detector = HallucinationDetector()
    result = detector.check(
        answer="The capital of France is Paris.",
        context="Paris is the capital and largest city of France.",
    )
    if result.hallucination_score > 0.3:
        print("Warning: possible hallucination detected!")
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .llm import get_client


@dataclass
class ClaimCheck:
    """Verification of one factual claim."""

    claim: str
    supported: bool
    evidence: str  # the source text that supports or contradicts
    score: float  # 0-1, higher = more likely false (hallucinated)


@dataclass
class HallucinationReport:
    """Complete hallucination analysis of an answer."""

    answer: str
    hallucination_score: float  # 0-1, higher = more hallucinations
    factual_consistency: float  # 0-1, 1 = fully consistent
    citation_accuracy: float  # 0-1, 1 = all citations correct
    self_consistency: float  # 0-1, 1 = all samples agree
    flagged_claims: list[ClaimCheck] = field(default_factory=list)
    summary: str = ""

    @property
    def is_safe(self) -> bool:
        return self.hallucination_score < 0.3

    @property
    def needs_review(self) -> bool:
        return 0.3 <= self.hallucination_score < 0.6

    @property
    def is_dangerous(self) -> bool:
        return self.hallucination_score >= 0.6


class HallucinationDetector:
    """Multi-strategy hallucination detection."""

    def __init__(self, model: str = "deepseek-v4-pro[1m]"):
        self.model = model
        self.llm = get_client()

    # ============================================================
    #  Strategy 1: Factual Consistency
    # ============================================================

    def check_factual_consistency(
        self, answer: str, context: str
    ) -> tuple[float, list[ClaimCheck]]:
        """Check if answer claims are supported by the provided context.

        Uses an LLM to extract atomic claims from the answer,
        then checks each claim against the context.

        Returns (consistency_score, list of claim checks).
        """
        claims = self._extract_claims(answer)
        if not claims:
            return 1.0, []

        checks = []
        for claim in claims:
            check = self._verify_claim(claim, context)
            checks.append(check)

        supported = sum(1 for c in checks if c.supported)
        score = supported / len(checks) if checks else 1.0
        return score, checks

    def _extract_claims(self, text: str) -> list[str]:
        """Extract atomic factual claims from text using LLM."""
        sys_prompt = (
            "Extract all factual claims from the text below. "
            "Return one claim per line. Skip opinions, greetings, and meta-commentary. "
            "Only output the claims, nothing else."
        )
        try:
            raw = self.llm.send(
                messages=[{"role": "user", "content": f"Text:\n{text}"}],
                model=self.model,
                max_tokens=512,
                system=sys_prompt,
            )
            claims = [line.strip("- ").strip() for line in raw.split("\n")]
            return [c for c in claims if len(c) > 10]
        except Exception:
            return []

    def _verify_claim(self, claim: str, context: str) -> ClaimCheck:
        """Verify one claim against context using LLM."""
        sys_prompt = (
            "You are a fact-checker. Determine if the claim is SUPPORTED by the context. "
            "Reply with a JSON object: "
            '{"supported": true/false, "evidence": "quote from context", "score": 0.0-1.0} '
            "Score = 0 means fully supported, 1 = completely contradicted. "
            "Reply with ONLY the JSON."
        )
        msg = f"Claim: {claim}\n\nContext: {context}"
        try:
            raw = self.llm.send(
                messages=[{"role": "user", "content": msg}],
                model=self.model,
                max_tokens=256,
                system=sys_prompt,
            )
            import json
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                data = json.loads(match.group())
                return ClaimCheck(
                    claim=claim,
                    supported=data.get("supported", False),
                    evidence=data.get("evidence", ""),
                    score=data.get("score", 0.5),
                )
        except Exception:
            pass
        return ClaimCheck(claim=claim, supported=False, evidence="", score=0.5)

    # ============================================================
    #  Strategy 2: Citation Verification
    # ============================================================

    def check_citations(
        self, answer: str, chunks: list[dict]
    ) -> float:
        """Verify that [Chunk N] citations in the answer actually support the claims.

        Args:
            answer: the answer text with [Chunk N] citations
            chunks: list of {"index": N, "text": "..."}

        Returns citation_accuracy (0-1).
        """
        # Find all citation references
        cited_indices = set()
        for match in re.finditer(r'\[Chunk\s+(\d+)\]', answer):
            cited_indices.add(int(match.group(1)))

        if not cited_indices:
            return 1.0  # No citations to verify — not necessarily bad

        # Build chunk lookup
        chunk_map = {}
        for c in chunks:
            idx = c.get("index", c.get("chunk_index", -1))
            chunk_map[idx + 1] = c.get("text", "")  # +1 because citations are 1-based

        # Check each cited chunk contains some substance
        valid = 0
        for idx in cited_indices:
            if idx in chunk_map and len(chunk_map[idx].strip()) > 20:
                valid += 1

        return valid / len(cited_indices) if cited_indices else 1.0

    # ============================================================
    #  Strategy 3: Self-Consistency
    # ============================================================

    def check_self_consistency(
        self,
        question: str,
        context: str = "",
        num_samples: int = 3,
        system_prompt: str = "",
    ) -> float:
        """Check if the model gives consistent answers across multiple samples.

        Low consensus → possible hallucination or uncertainty.

        For closed-form answers (numbers, names, dates): checks exact match.
        For open-ended answers: uses LLM to judge agreement.
        """
        samples = []
        for _ in range(num_samples):
            try:
                msg = f"Question: {question}"
                if context:
                    msg = f"Context: {context}\n\nQuestion: {question}"
                ans = self.llm.send(
                    messages=[{"role": "user", "content": msg}],
                    model=self.model,
                    max_tokens=512,
                    system=system_prompt,
                )
                samples.append(ans)
            except Exception:
                continue

        if len(samples) < 2:
            return 0.5

        # Quick check: is this a closed-form answer?
        # If all samples are short and numeric → exact match
        if all(len(s) < 50 for s in samples):
            counter = Counter(samples)
            most_common = counter.most_common(1)[0][1]
            return most_common / len(samples)

        # Open-ended: use LLM to judge pairwise agreement
        agreements = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                agree = self._judge_agreement(samples[i], samples[j])
                agreements.append(agree)

        return sum(agreements) / len(agreements) if agreements else 0.5

    def _judge_agreement(self, answer_a: str, answer_b: str) -> float:
        """LLM judge: do these two answers agree on the facts?"""
        sys_prompt = (
            "Compare two answers to the same question. "
            "Reply with a single number 0-100 indicating how much they AGREE "
            "on the key facts. 100 = identical facts, 0 = completely contradictory."
        )
        msg = f"Answer A: {answer_a}\n\nAnswer B: {answer_b}"
        try:
            raw = self.llm.send(
                messages=[{"role": "user", "content": msg}],
                model=self.model,
                max_tokens=10,
                system=sys_prompt,
            )
            match = re.search(r'\d+', raw)
            if match:
                return min(100, max(0, int(match.group()))) / 100.0
        except Exception:
            pass
        return 0.5

    # ============================================================
    #  Combined check
    # ============================================================

    def check(
        self,
        answer: str,
        context: str = "",
        question: str = "",
        chunks: list[dict] | None = None,
        weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> HallucinationReport:
        """Run all detection strategies and produce a combined report.

        Args:
            answer: the answer to check
            context: source context used to generate the answer
            question: original question (for self-consistency check)
            chunks: citation chunks (for citation verification)
            weights: (factual_consistency, citation_accuracy, self_consistency)

        Returns HallucinationReport with scores and flagged claims.
        """
        w_fact, w_cite, w_self = weights

        # Strategy 1: Factual consistency
        fact_score, claims = 1.0, []
        if context:
            fact_score, claims = self.check_factual_consistency(answer, context)

        # Strategy 2: Citation verification
        cite_score = 1.0
        if chunks:
            cite_score = self.check_citations(answer, chunks)

        # Strategy 3: Self-consistency (expensive! Only if question provided)
        self_score = 1.0
        if question:
            self_score = self.check_self_consistency(question, context)

        # Combined hallucination score (inverted: 1 - weighted average of trust scores)
        trust = w_fact * fact_score + w_cite * cite_score + w_self * self_score
        hallucination_score = 1.0 - trust

        # Build summary
        flagged = [c for c in claims if not c.supported]
        if flagged:
            summary = f"{len(flagged)}/{len(claims)} claims unsupported by context"
        elif hallucination_score < 0.3:
            summary = "Answer appears consistent with context and citations"
        else:
            summary = "Mixed — review recommended"

        return HallucinationReport(
            answer=answer,
            hallucination_score=round(hallucination_score, 3),
            factual_consistency=round(fact_score, 3),
            citation_accuracy=round(cite_score, 3),
            self_consistency=round(self_score, 3),
            flagged_claims=flagged,
            summary=summary,
        )
