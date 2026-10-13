"""Deterministic safety and escalation policy independent of the LLM."""
import re
from dataclasses import dataclass

from src.agent.models import AgentEscalationReason, SupportRequest
from src.data.intent_taxonomy import Intent
from src.knowledge.retriever import RetrievalResult


@dataclass(frozen=True)
class EscalationOutcome:
    should_escalate: bool
    reason: AgentEscalationReason


class EscalationPolicy:
    """Conservative rules for deciding whether a draft may be auto-handled."""

    HIGH_RISK_PATTERNS = (
        r"\b(?:lawsuit|lawyer|legal action|police|regulator)\b",
        r"\b(?:injury|injured|fire|smoke|explod|electric shock)\w*\b",
        r"\b(?:fraud|stolen card|identity theft|hacked|compromised)\b",
        r"\b(?:threat|kill|suicide|self[- ]harm)\w*\b",
    )
    TRANSACTION_PATTERNS = (
        r"\b(?:my|this|the)\s+(?:order|refund|return|replacement|charge)\b",
        r"\b(?:charged|debited|refund pending|money back)\b",
        r"\b(?:cancel|close)\s+(?:my|the)\b",
        r"\b(?:where is|track)\s+my\b",
    )
    REPEATED_PATTERNS = (
        r"\b(?:again|still|multiple times|third time|keeps happening)\b",
        r"\b(?:no response|not resolved|never fixed|already contacted)\b",
    )
    UNSUPPORTED_CLAIM_PATTERNS = (
        r"\b(?:we|i)(?:'ve| have)?\s+(?:issued|processed|approved|sent|cancelled|closed)\b",
        r"\b(?:your refund has|your replacement has|your order will)\b",
        r"\b(?:will arrive|will be delivered|guarantee|confirmed for)\b",
        r"\b(?:i can see|our records show|your account shows)\b",
    )

    @staticmethod
    def _matches(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def evaluate(
        self,
        request: SupportRequest,
        intent: Intent,
        retrieval: RetrievalResult,
    ) -> EscalationOutcome:
        combined_text = " ".join([*request.conversation_history, request.message])
        if self._matches(combined_text, self.HIGH_RISK_PATTERNS):
            return EscalationOutcome(True, AgentEscalationReason.HIGH_RISK)
        if intent == Intent.ACCOUNT_ACCESS_SECURITY:
            return EscalationOutcome(
                True,
                AgentEscalationReason.ACCOUNT_VERIFICATION_REQUIRED,
            )
        if intent == Intent.OTHER_UNKNOWN:
            return EscalationOutcome(True, AgentEscalationReason.UNSUPPORTED_INTENT)
        if len(request.conversation_history) >= 2 and self._matches(
            combined_text,
            self.REPEATED_PATTERNS,
        ):
            return EscalationOutcome(
                True,
                AgentEscalationReason.REPEATED_UNRESOLVED,
            )
        if self._matches(request.message, self.TRANSACTION_PATTERNS):
            return EscalationOutcome(
                True,
                AgentEscalationReason.TRANSACTION_VERIFICATION_REQUIRED,
            )
        if retrieval.is_empty:
            return EscalationOutcome(
                True,
                AgentEscalationReason.INSUFFICIENT_EVIDENCE,
            )
        return EscalationOutcome(False, AgentEscalationReason.NONE)

    def contains_unsupported_claim(self, reply: str) -> bool:
        return self._matches(reply, self.UNSUPPORTED_CLAIM_PATTERNS)
