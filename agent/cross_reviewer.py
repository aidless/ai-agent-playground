"""Cross-Model Reviewer — 跨模型校对官

Implements the workflow described by Liu Zewen:
  1. A main agent (Claude/DeepSeek) generates a report
  2. A reviewer agent (different model: Qwen/MiniMax) verifies every claim
  3. Disputed items enter a debate loop until consensus or deadlock
  4. Unresolved items escalate to human for final decision

Key insight: different models have different "perspectives" (training data,
architecture, reasoning style). This makes cross-model review far more
effective than same-model self-review — just like two humans catch each
other's blind spots.

Architecture:
    Main Agent (DeepSeek)  ──generates──▶  Report
                                              │
    Reviewer (Qwen2.5)    ──checks────▶  Findings
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                              Consensus Items      Disputed Items
                                    │                    │
                                    ▼                    ▼
                              Auto-fix            Debate Loop
                                                       │
                                               ┌───────┴───────┐
                                               ▼               ▼
                                          Resolved        Escalate to
                                                          Human
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────

class FindingType(str, Enum):
    FACTUAL_ERROR = "factual_error"       # 引用错误 — 条文不存在或内容不对
    SEMANTIC_ERROR = "semantic_error"     # 语义错误 — 理解偏差
    OMISSION = "omission"                 # 遗漏 — 应该提到但没提
    INCONSISTENCY = "inconsistency"       # 前后矛盾
    SUGGESTION = "suggestion"             # 改进建议


class FindingStatus(str, Enum):
    OPEN = "open"
    ACCEPTED = "accepted"         # Original author accepted the finding
    REJECTED = "rejected"         # Original author rejected the finding
    DEBATING = "debating"         # Currently in debate
    CONSENSUS = "consensus"       # Both sides agreed
    ESCALATED = "escalated"       # Sent to human for decision
    RESOLVED = "resolved"         # Final resolution applied


@dataclass
class ReviewFinding:
    """One issue found by the reviewer."""
    id: str
    type: FindingType
    location: str = ""              # Where in the original text
    claim: str = ""                 # What the original says
    finding: str = ""              # What the reviewer says is wrong
    evidence: str = ""             # Supporting evidence from source
    suggestion: str = ""           # How to fix it
    status: FindingStatus = FindingStatus.OPEN
    author_response: str = ""      # Original author's rebuttal
    reviewer_counter: str = ""     # Reviewer's counter to rebuttal
    final_resolution: str = ""     # Human decision if escalated
    severity: str = "medium"       # critical / high / medium / low
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ReviewResult:
    """Complete review result."""
    review_id: str
    original_text: str
    findings: list[ReviewFinding] = field(default_factory=list)
    accepted_count: int = 0
    rejected_count: int = 0
    escalated_count: int = 0
    consensus_count: int = 0
    total_debate_rounds: int = 0
    final_text: str = ""
    completed: bool = False


# ── Reviewer prompts ──────────────────────────────

REVIEWER_SYSTEM = """你是严格的校对官。逐条检查报告的每个论断。

对每个发现的问题，严格按以下格式输出（这是强制格式，必须遵守）:

[FINDING]
TYPE: factual_error
LOCATION: 原文原句
ISSUE: 问题是什么
EVIDENCE: 证据或依据
FIX: 建议修改为
[/FINDING]

TYPE 必须是以下之一: factual_error / semantic_error / omission / inconsistency / suggestion
如果没有发现问题，输出: NO_FINDINGS"""


DEBATE_PROMPT = """The original author has responded to your review finding. They either accepted it
(no further action needed) or rejected it with a rebuttal. If they rejected it, review their rebuttal:

Original claim: {claim}
Your finding: {finding}
Author's rebuttal: {rebuttal}

Consider:
1. Is the author's rebuttal valid? Did they cite evidence?
2. Is there a misunderstanding on either side?
3. Can you find common ground — a version of the correction both sides would accept?

Respond with:
[VERDICT]
ACCEPT_REBUTTAL | STAND_BY_FINDING | COMPROMISE
COMPROMISE_TEXT: <middle-ground wording that both sides could accept>
REASON: <why you chose this verdict>
[/VERDICT]"""


CONSENSUS_PROMPT = """You are a neutral mediator. Two agents disagree on the following:

Claim: {claim}
Reviewer says: {finding}
Author says: {rebuttal}
Reviewer's final position: {counter}

Propose a resolution that both sides can accept. If truly irreconcilable, mark as ESCALATE.
Respond with:
[RESOLUTION]
CONSENSUS | ESCALATE
TEXT: <agreed text or compromise>
[/RESOLUTION]"""


# ── CrossReviewer ──────────────────────────────────

class CrossReviewer:
    """跨模型校对系统

    Usage:
        reviewer = CrossReviewer(
            primary_client=deepseek_client,    # Main model
            reviewer_client=ollama_client,     # Different model for review
            primary_model="deepseek-chat",
            reviewer_model="qwen2.5:7b",
        )
        result = await reviewer.review(report_text, source_context="")
    """

    def __init__(
        self,
        primary_client,           # Main agent's LLM client (DeepSeek)
        reviewer_client,          # Reviewer's LLM client (Qwen/MiniMax — DIFFERENT model)
        primary_model: str = "deepseek-chat",
        reviewer_model: str = "qwen2.5:7b",
        max_debate_rounds: int = 2,
    ):
        self.primary = primary_client
        self.reviewer = reviewer_client
        self.primary_model = primary_model
        self.reviewer_model = reviewer_model
        self.max_debate_rounds = max_debate_rounds
        self._results: dict[str, ReviewResult] = {}

    async def review(
        self,
        original_text: str,
        source_context: str = "",
        instructions: str = "",
    ) -> ReviewResult:
        """Perform a full cross-model review.

        Args:
            original_text: The report/text to review
            source_context: Any source documents the report references
            instructions: Special review instructions
        """
        review_id = hashlib.md5(
            f"{original_text[:100]}{time.time()}".encode()
        ).hexdigest()[:12]

        result = ReviewResult(
            review_id=review_id,
            original_text=original_text,
        )
        self._results[review_id] = result

        # Phase 1: Reviewer checks every claim
        logger.info("[Review %s] Phase 1: Initial review", review_id)
        findings = await self._phase1_review(original_text, source_context, instructions)
        result.findings = findings

        if not findings:
            result.completed = True
            result.final_text = original_text
            logger.info("[Review %s] No issues found — clean!", review_id)
            return result

        logger.info("[Review %s] Found %d issues", review_id, len(findings))

        # Phase 2: Author responds to each finding
        logger.info("[Review %s] Phase 2: Author response", review_id)
        await self._phase2_author_respond(result, original_text, source_context)

        # Phase 3: Debate disputed items
        disputed = [f for f in result.findings if f.status == FindingStatus.DEBATING]
        if disputed:
            logger.info("[Review %s] Phase 3: Debate %d disputed items", review_id, len(disputed))
            await self._phase3_debate(result, disputed, original_text, source_context)

        # Phase 4: Apply fixes and produce final text
        logger.info("[Review %s] Phase 4: Apply fixes", review_id)
        result.final_text = await self._phase4_apply(result)

        # Count results
        result.accepted_count = sum(1 for f in result.findings if f.status == FindingStatus.ACCEPTED)
        result.rejected_count = sum(1 for f in result.findings if f.status == FindingStatus.REJECTED)
        result.escalated_count = sum(1 for f in result.findings if f.status == FindingStatus.ESCALATED)
        result.consensus_count = sum(1 for f in result.findings if f.status == FindingStatus.CONSENSUS)
        result.completed = True

        logger.info(
            "[Review %s] Complete: %d accepted, %d rejected, %d consensus, %d escalated",
            review_id, result.accepted_count, result.rejected_count,
            result.consensus_count, result.escalated_count,
        )

        return result

    async def _phase1_review(
        self, text: str, source: str, instructions: str
    ) -> list[ReviewFinding]:
        """Reviewer (different model) checks the text."""
        prompt = f"""Review the following report. Check every factual claim, every reference,
and every conclusion against the provided source context.

SOURCE CONTEXT:
{source[:8000] if source else "No source context provided — check for internal consistency and logical errors."}

SPECIAL INSTRUCTIONS:
{instructions if instructions else "Check all claims for accuracy, completeness, and consistency."}

REPORT TO REVIEW:
{text[:12000]}

{REVIEWER_SYSTEM}"""

        try:
            response = await self.reviewer.chat.completions.create(
                model=self.reviewer_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.1,  # Low temperature for consistent review
            )
            reviewer_output = response.choices[0].message.content
            logger.debug("[Review] Raw reviewer output (%d chars): %s...",
                        len(reviewer_output), reviewer_output[:300])
        except Exception as e:
            logger.error("[Review] Reviewer LLM call failed: %s", e)
            return []

        # Parse findings from reviewer output
        findings = self._parse_findings(reviewer_output)
        if not findings:
            logger.warning("[Review] No findings parsed. Reviewer raw: %s",
                          reviewer_output[:500] if reviewer_output else "(empty)")
        return findings

    def _parse_findings(self, output: str) -> list[ReviewFinding]:
        """Parse findings from reviewer output. Handles multiple formats."""
        findings = []

        # Method 1: [FINDING]...[/FINDING] blocks (case-insensitive detection)
        if "[FINDING]" in output.upper():
            # Split on original case-aware blocks
            import re
            blocks = re.split(r'\[FINDING\]', output, flags=re.IGNORECASE)
            for i, block in enumerate(blocks):
                if i == 0:
                    continue  # Skip text before first [FINDING]
                # Find closing [/FINDING]
                end_match = re.search(r'\[/FINDING\]', block, flags=re.IGNORECASE)
                if end_match:
                    content = block[:end_match.start()].strip()
                else:
                    content = block.strip()
                    if len(content) < 20:
                        continue

                fields = {}
                for line in content.split("\n"):
                    line = line.strip()
                    if ":" not in line:
                        continue
                    raw_key = line.split(":", 1)[0].strip().lower()
                    raw_val = line.split(":", 1)[1].strip()
                    if raw_key in ["type", "location", "issue", "evidence", "fix"]:
                        fields[raw_key] = raw_val

                ftype_str = fields.get("type", "suggestion")
                try:
                    ftype = FindingType(ftype_str)
                except ValueError:
                    ftype = FindingType.SUGGESTION

                finding = ReviewFinding(
                    id=f"F{len(findings):03d}",
                    type=ftype,
                    location=fields.get("location", ""),
                    claim=fields.get("location", "")[:200],
                    finding=fields.get("issue", ""),
                    evidence=fields.get("evidence", ""),
                    suggestion=fields.get("fix", ""),
                )
                if finding.finding:
                    findings.append(finding)

        # Method 2: Numbered/bullet list format (fallback for smaller models)
        if not findings:
            findings = self._parse_list_format(output)

        return findings

    def _parse_list_format(self, output: str) -> list[ReviewFinding]:
        """Parse numbered/bullet list format common with smaller models."""
        findings = []
        lines = output.split("\n")
        current = {}
        in_finding = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_finding and current.get("finding"):
                    findings.append(ReviewFinding(
                        id=f"F{len(findings):03d}",
                        type=FindingType(current.get("type", "suggestion")),
                        location=current.get("location", ""),
                        claim=current.get("location", "")[:200],
                        finding=current.get("finding", ""),
                        evidence=current.get("evidence", ""),
                        suggestion=current.get("suggestion", ""),
                    ))
                    current = {}
                    in_finding = False
                continue

            # Detect finding start: numbered items, "Issue", "Finding", "Error", "问题", "错误"
            starts = ["错误", "问题", "遗漏", "不一致", "建议",
                      "ERROR", "ISSUE", "FINDING", "OMISSION",
                      "error:", "issue:", "finding:", "problem:"]
            if any(line.upper().startswith(s.upper()) for s in starts) or \
               (len(line) > 0 and line[0].isdigit() and ("." in line[:4] or ")" in line[:4] or "、" in line[:4])):
                if in_finding and current.get("finding"):
                    findings.append(ReviewFinding(
                        id=f"F{len(findings):03d}",
                        type=FindingType(current.get("type", "suggestion")),
                        location=current.get("location", ""),
                        claim=current.get("location", "")[:200],
                        finding=current.get("finding", ""),
                        evidence=current.get("evidence", ""),
                        suggestion=current.get("suggestion", ""),
                    ))
                current = {}
                in_finding = True
                current["finding"] = line

            elif in_finding:
                # Classify the line
                lower = line.lower()
                if any(w in lower for w in ["原文", "location", "出处", "位置"]):
                    current["location"] = line.split(":", 1)[-1].strip() if ":" in line else line
                elif any(w in lower for w in ["证据", "依据", "evidence", "source", "参考"]):
                    current["evidence"] = line.split(":", 1)[-1].strip() if ":" in line else line
                elif any(w in lower for w in ["修改", "建议", "fix", "suggest", "改为", "更正"]):
                    current["suggestion"] = line.split(":", 1)[-1].strip() if ":" in line else line
                elif any(w in lower for w in ["factual", "semantic", "omission", "inconsistency"]):
                    current["type"] = line.split(":")[0].strip().lower()
                elif "type" in lower:
                    current["type"] = line.split(":", 1)[-1].strip().lower().replace(" ", "_")
                else:
                    current["finding"] = (current.get("finding", "") + " " + line).strip()

        # Last finding
        if in_finding and current.get("finding"):
            findings.append(ReviewFinding(
                id=f"F{len(findings):03d}",
                type=FindingType(current.get("type", "suggestion")),
                location=current.get("location", ""),
                claim=current.get("location", "")[:200],
                finding=current.get("finding", ""),
                evidence=current.get("evidence", ""),
                suggestion=current.get("suggestion", ""),
            ))

        return findings

    async def _phase2_author_respond(
        self, result: ReviewResult, original: str, source: str
    ):
        """Original author (primary model) responds to each finding."""
        for finding in result.findings:
            prompt = f"""你是这份报告的原作者。有人校对了你的报告并提出了问题。

【你的原文】
{original[:3000]}

【校对意见】
类型: {finding.type.value}
出处: {finding.location}
问题: {finding.finding}
依据: {finding.evidence}
建议: {finding.suggestion}

【你的回应】
你有三个选择：
- ACCEPT：校对完全正确，接受修改
- REJECT：校对是错的，你的原文更正确。你必须用证据反驳
- DEBATE：校对有部分道理，但你也有合理之处。说出你的立场

注意：不要轻易接受！如果原文有合理依据、或者校对官的批评过于苛刻、
或者这是一个判断问题而非事实问题——你应该坚持自己的立场或进入辩论。

格式：
[RESPONSE]
ACCEPT | REJECT | DEBATE
理由: <你的论证>
[/RESPONSE]"""

            try:
                response = await self.primary.chat.completions.create(
                    model=self.primary_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1,
                )
                resp_text = response.choices[0].message.content
                resp_upper = resp_text.upper()

                if "REJECT" in resp_upper and "ACCEPT" not in resp_upper:
                    finding.status = FindingStatus.REJECTED
                elif "DEBATE" in resp_upper or "DEBATING" in resp_upper:
                    finding.status = FindingStatus.DEBATING
                elif "ACCEPT" in resp_upper:
                    finding.status = FindingStatus.ACCEPTED
                else:
                    finding.status = FindingStatus.ACCEPTED  # Default accept

                # Extract explanation
                for marker in ["理由:", "EXPLANATION:", "解释:", "说明:"]:
                    if marker in resp_text:
                        finding.author_response = resp_text.split(marker, 1)[1].split("[/RESPONSE]")[0].strip()[:400]
                        break
                if not finding.author_response:
                    finding.author_response = resp_text[:400]

            except Exception as e:
                logger.error("[Review] Author response failed for %s: %s", finding.id, e)
                finding.status = FindingStatus.ESCALATED

    async def _phase3_debate(
        self, result: ReviewResult, disputed: list[ReviewFinding],
        original: str, source: str,
    ):
        """Debate loop: reviewer counters author's rejection, then mediate."""
        for round_num in range(self.max_debate_rounds):
            still_disputed = [f for f in disputed if f.status == FindingStatus.DEBATING]
            if not still_disputed:
                break

            result.total_debate_rounds = round_num + 1

            for finding in still_disputed:
                # Reviewer counters
                prompt = DEBATE_PROMPT.format(
                    claim=finding.claim,
                    finding=finding.finding,
                    rebuttal=finding.author_response,
                )

                try:
                    response = await self.reviewer.chat.completions.create(
                        model=self.reviewer_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.1,
                    )
                    counter = response.choices[0].message.content
                    finding.reviewer_counter = counter[:500]

                    if "ACCEPT_REBUTTAL" in counter:
                        finding.status = FindingStatus.REJECTED
                    elif "COMPROMISE" in counter:
                        finding.suggestion = counter.split("COMPROMISE_TEXT:")[1].split("\n")[0].strip() if "COMPROMISE_TEXT:" in counter else finding.suggestion
                        finding.status = FindingStatus.CONSENSUS
                    else:
                        finding.status = FindingStatus.DEBATING  # Stay in debate
                except Exception as e:
                    logger.error("[Review] Debate failed for %s: %s", finding.id, e)
                    finding.status = FindingStatus.ESCALATED

        # Final pass: items still disputed → escalate
        for finding in disputed:
            if finding.status == FindingStatus.DEBATING:
                finding.status = FindingStatus.ESCALATED

    async def _phase4_apply(self, result: ReviewResult) -> str:
        """Apply accepted and consensus fixes to produce final text."""
        text = result.original_text

        accepted_fixes = [
            f for f in result.findings
            if f.status in (FindingStatus.ACCEPTED, FindingStatus.CONSENSUS)
            and f.suggestion
        ]

        if not accepted_fixes:
            return text

        # Build a fix application prompt
        fixes_text = "\n".join(
            f"- Original: \"{f.location}\"\n  Fix: {f.suggestion}"
            for f in accepted_fixes
        )

        prompt = f"""Apply the following corrections to the text below. Return the corrected text.
Make ONLY the specified changes. Do not add or remove anything else.

CORRECTIONS:
{fixes_text[:3000]}

ORIGINAL TEXT:
{text[:10000]}"""

        try:
            response = await self.primary.chat.completions.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("[Review] Fix application failed: %s", e)
            return text  # Return original on failure

    def get_result(self, review_id: str) -> ReviewResult | None:
        return self._results.get(review_id)

    def get_escalated(self, review_id: str) -> list[ReviewFinding]:
        """Get escalated items that need human decision."""
        result = self._results.get(review_id)
        if not result:
            return []
        return [f for f in result.findings if f.status == FindingStatus.ESCALATED]

    def resolve_escalated(self, review_id: str, finding_id: str, decision: str, resolution: str = ""):
        """Human resolves an escalated finding."""
        result = self._results.get(review_id)
        if not result:
            return
        for f in result.findings:
            if f.id == finding_id:
                f.final_resolution = decision
                f.status = FindingStatus.RESOLVED
                if resolution:
                    f.suggestion = resolution
                break


# ── Factory for Ollama client ──────────────────────

def create_ollama_reviewer_client(base_url: str = "http://localhost:11434/v1") -> AsyncOpenAI:
    """Create an OpenAI-compatible client pointing to local Ollama.

    Ollama serves an OpenAI-compatible API at http://localhost:11434/v1
    since version 0.24+. Qwen2.5:7b works well as a reviewer model.
    """
    return AsyncOpenAI(
        api_key="ollama",  # Ollama doesn't require a real key
        base_url=base_url,
    )
