"""Intent taxonomy definition and Golden Evaluation Example schema."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Intent(str, Enum):
    """Defensible customer intent categories derived from Amazon customer support interactions."""
    ORDER_STATUS_DELIVERY = "order_status_delivery"
    RETURNS_REFUNDS = "returns_refunds"
    DAMAGED_DEFECTIVE_ITEM = "damaged_defective_item"
    ACCOUNT_ACCESS_SECURITY = "account_access_security"
    SUBSCRIPTION_DIGITAL = "subscription_digital"
    PRODUCT_INQUIRY_AVAILABILITY = "product_inquiry_availability"
    OTHER_UNKNOWN = "other_unknown"

    @classmethod
    def list_values(cls) -> List[str]:
        return [item.value for item in cls]


class Action(str, Enum):
    AUTO_HANDLE = "auto_handle"
    ESCALATE = "escalate"


class EscalationReason(str, Enum):
    ACCOUNT_SECURITY_PII = "account_security_pii"
    ORDER_LOOKUP_REQUIRED = "order_lookup_required"
    REFUND_REPLACEMENT_DISPUTE = "refund_replacement_dispute"
    AMBIGUOUS_UNSUPPORTED = "ambiguous_unsupported"
    HIGH_SENTIMENT_RISK = "high_sentiment_risk"
    NONE = "none"


class GoldenEvaluationExample(BaseModel):
    """Schema for hand-verified ground truth evaluation examples."""
    id: str = Field(description="Unique identifier for evaluation example")
    customer_tweet_id: int
    customer_author_id: str
    customer_message: str
    cleaned_customer_message: str
    conversation_context: Optional[str] = None
    expected_intent: Intent
    expected_action: Action
    escalation_reason: EscalationReason = EscalationReason.NONE
    annotation_notes: Optional[str] = None
    historical_brand_reply: Optional[str] = None
    is_synthetic: bool = False

    @field_validator("escalation_reason")
    @classmethod
    def validate_escalation_reason(cls, v: EscalationReason, info):
        # If action is auto_handle, reason must be none; if escalate, reason must not be none
        action = info.data.get("expected_action")
        if action == Action.AUTO_HANDLE and v != EscalationReason.NONE:
            raise ValueError("escalation_reason must be EscalationReason.NONE when expected_action is auto_handle")
        if action == Action.ESCALATE and v == EscalationReason.NONE:
            raise ValueError("escalation_reason must specify a valid reason when expected_action is escalate")
        return v
