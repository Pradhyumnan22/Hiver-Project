"""Validated inputs and outputs for the support-agent pipeline."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.data.intent_taxonomy import Action, Intent


class AgentEscalationReason(str, Enum):
    NONE = "none"
    HIGH_RISK = "high_risk"
    UNSUPPORTED_INTENT = "unsupported_intent"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ACCOUNT_VERIFICATION_REQUIRED = "account_verification_required"
    TRANSACTION_VERIFICATION_REQUIRED = "transaction_verification_required"
    REPEATED_UNRESOLVED = "repeated_unresolved"
    MODEL_FAILURE = "model_failure"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class SupportRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_history: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value


class IntentClassification(BaseModel):
    intent: Intent
    rationale: str = Field(min_length=3, max_length=500)
    uncertainty: Optional[str] = Field(default=None, max_length=500)


class GroundedReplyDraft(BaseModel):
    reply: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    uncertainty: Optional[str] = Field(default=None, max_length=500)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("evidence_ids must be unique")
        return values


class AgentDecision(BaseModel):
    action: Action
    reason: AgentEscalationReason
    reason_detail: str = Field(min_length=3, max_length=500)
    intent: Intent
    evidence_ids: list[str] = Field(default_factory=list)
    proposed_reply: Optional[str] = None
    uncertainty: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AgentDecision":
        if self.action == Action.AUTO_HANDLE:
            if self.reason != AgentEscalationReason.NONE:
                raise ValueError("auto_handle decisions must use reason=none")
            if not self.proposed_reply:
                raise ValueError("auto_handle decisions require a proposed reply")
            if not self.evidence_ids:
                raise ValueError("auto_handle decisions require evidence IDs")
        elif self.reason == AgentEscalationReason.NONE:
            raise ValueError("escalations require a reason")
        return self
