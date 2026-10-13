"""Single-call grounded support-agent orchestration."""
from src.agent.escalation import EscalationPolicy
from src.agent.llm import LLMResponseError, StructuredLLM
from src.agent.models import (
    AgentDecision,
    AgentEscalationReason,
    GroundedReplyDraft,
    IntentClassification,
    SupportRequest,
)
from src.agent.prompts import (
    INTENT_SYSTEM_PROMPT,
    REPLY_SYSTEM_PROMPT,
    build_intent_prompt,
    build_reply_prompt,
)
from src.data.intent_taxonomy import Action, Intent
from src.knowledge.retriever import FAISSRetriever

REASON_DETAILS = {
    AgentEscalationReason.NONE: "Evidence and deterministic safety checks support a draft reply.",
    AgentEscalationReason.HIGH_RISK: "The message contains legal, fraud, safety, or severe-harm risk.",
    AgentEscalationReason.UNSUPPORTED_INTENT: "The message is ambiguous or outside the supported intent taxonomy.",
    AgentEscalationReason.INSUFFICIENT_EVIDENCE: "No historical evidence met the calibrated relevance threshold.",
    AgentEscalationReason.ACCOUNT_VERIFICATION_REQUIRED: "The issue requires secure account or identity verification.",
    AgentEscalationReason.TRANSACTION_VERIFICATION_REQUIRED: "The request requires order, payment, refund, or account-specific verification.",
    AgentEscalationReason.REPEATED_UNRESOLVED: "Conversation history indicates repeated unsuccessful support attempts.",
    AgentEscalationReason.MODEL_FAILURE: "A required model or retrieval component failed validation.",
    AgentEscalationReason.UNSUPPORTED_CLAIM: "The generated draft contained an unsupported action, status, or outcome claim.",
}


class SupportAgent:
    """Classify, retrieve, enforce policy, and draft using validated outputs."""

    def __init__(
        self,
        *,
        llm: StructuredLLM,
        retriever: FAISSRetriever,
        escalation_policy: EscalationPolicy | None = None,
        top_k: int = 3,
    ):
        self.llm = llm
        self.retriever = retriever
        self.escalation_policy = escalation_policy or EscalationPolicy()
        self.top_k = top_k

    @staticmethod
    def _escalation(
        *,
        reason: AgentEscalationReason,
        intent: Intent,
        evidence_ids: list[str] | None = None,
        uncertainty: str | None = None,
    ) -> AgentDecision:
        return AgentDecision(
            action=Action.ESCALATE,
            reason=reason,
            reason_detail=REASON_DETAILS[reason],
            intent=intent,
            evidence_ids=evidence_ids or [],
            proposed_reply=None,
            uncertainty=uncertainty,
        )

    def run(self, request: SupportRequest) -> AgentDecision:
        try:
            classification = self.llm.generate(
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_prompt=build_intent_prompt(
                    request.message,
                    request.conversation_history,
                ),
                output_model=IntentClassification,
            )
        except (LLMResponseError, ValueError, TypeError) as exc:
            return self._escalation(
                reason=AgentEscalationReason.MODEL_FAILURE,
                intent=Intent.OTHER_UNKNOWN,
                uncertainty=str(exc),
            )

        try:
            retrieval = self.retriever.retrieve(request.message, top_k=self.top_k)
        except Exception as exc:
            return self._escalation(
                reason=AgentEscalationReason.MODEL_FAILURE,
                intent=classification.intent,
                uncertainty=f"Retrieval failed: {exc}",
            )

        policy_outcome = self.escalation_policy.evaluate(
            request,
            classification.intent,
            retrieval,
        )
        if policy_outcome.should_escalate:
            return self._escalation(
                reason=policy_outcome.reason,
                intent=classification.intent,
                evidence_ids=retrieval.source_ids,
                uncertainty=classification.uncertainty,
            )

        try:
            draft = self.llm.generate(
                system_prompt=REPLY_SYSTEM_PROMPT,
                user_prompt=build_reply_prompt(
                    request.message,
                    request.conversation_history,
                    classification.intent,
                    retrieval,
                ),
                output_model=GroundedReplyDraft,
            )
        except (LLMResponseError, ValueError, TypeError) as exc:
            return self._escalation(
                reason=AgentEscalationReason.MODEL_FAILURE,
                intent=classification.intent,
                evidence_ids=retrieval.source_ids,
                uncertainty=str(exc),
            )

        supplied_ids = set(retrieval.source_ids)
        if not set(draft.evidence_ids).issubset(supplied_ids):
            return self._escalation(
                reason=AgentEscalationReason.UNSUPPORTED_CLAIM,
                intent=classification.intent,
                evidence_ids=retrieval.source_ids,
                uncertainty="Draft cited evidence that was not supplied.",
            )
        if self.escalation_policy.contains_unsupported_claim(draft.reply):
            return self._escalation(
                reason=AgentEscalationReason.UNSUPPORTED_CLAIM,
                intent=classification.intent,
                evidence_ids=draft.evidence_ids,
                uncertainty=draft.uncertainty,
            )

        return AgentDecision(
            action=Action.AUTO_HANDLE,
            reason=AgentEscalationReason.NONE,
            reason_detail=REASON_DETAILS[AgentEscalationReason.NONE],
            intent=classification.intent,
            evidence_ids=draft.evidence_ids,
            proposed_reply=draft.reply,
            uncertainty=draft.uncertainty or classification.uncertainty,
        )


def run_support_agent(
    request: SupportRequest,
    *,
    llm: StructuredLLM,
    retriever: FAISSRetriever,
) -> AgentDecision:
    """Convenience function exposing one callable pipeline entry point."""
    return SupportAgent(llm=llm, retriever=retriever).run(request)
