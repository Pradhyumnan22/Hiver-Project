"""Deterministic unit and integration tests for the support-agent pipeline."""
from collections import deque

from src.agent.escalation import EscalationPolicy
from src.agent.llm import LLMResponseError
from src.agent.models import (
    AgentEscalationReason,
    GroundedReplyDraft,
    IntentClassification,
    SupportRequest,
)
from src.agent.pipeline import SupportAgent
from src.data.intent_taxonomy import Action
from src.knowledge.retriever import HistoricalRecord, RetrievalResult


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = []

    def generate(self, *, system_prompt, user_prompt, output_model):
        self.calls.append(output_model)
        output = self.outputs.popleft()
        if isinstance(output, Exception):
            raise output
        return output_model.model_validate(output)


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query, top_k=3):
        self.calls.append((query, top_k))
        return self.result


def retrieval_result(*, empty=False, score=0.82):
    if empty:
        return RetrievalResult(
            query="query",
            records=[],
            scores=[],
            source_ids=[],
            is_empty=True,
            relevance_threshold=0.65,
            best_score=0.4,
        )
    record = HistoricalRecord(
        pair_id="case_1",
        customer_tweet_id=1,
        customer_author_id="historical_user",
        customer_text="Is this item compatible with my phone?",
        cleaned_customer_text="Is this item compatible with my phone?",
        historical_brand_reply="Please check the compatibility details on the product page.",
        cleaned_brand_reply="Please check compatibility details on the product page.",
        intent="product_inquiry_availability",
    )
    return RetrievalResult(
        query="query",
        records=[record],
        scores=[score],
        source_ids=["case_1"],
        is_empty=False,
        relevance_threshold=0.65,
        best_score=score,
    )


def classification(intent="product_inquiry_availability"):
    return {
        "intent": intent,
        "rationale": "The customer asks for public product information.",
        "uncertainty": None,
    }


def test_integration_auto_handles_grounded_safe_reply():
    llm = FakeLLM(
        [
            classification(),
            {
                "reply": "Please check the product details page for compatibility information.",
                "evidence_ids": ["case_1"],
                "uncertainty": "Compatibility can vary by model.",
            },
        ]
    )
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(SupportRequest(message="Is this case compatible with my phone?"))

    assert decision.action == Action.AUTO_HANDLE
    assert decision.reason == AgentEscalationReason.NONE
    assert decision.evidence_ids == ["case_1"]
    assert decision.proposed_reply
    assert llm.calls == [IntentClassification, GroundedReplyDraft]


def test_insufficient_evidence_escalates_before_generation():
    llm = FakeLLM([classification()])
    agent = SupportAgent(
        llm=llm,
        retriever=FakeRetriever(retrieval_result(empty=True)),
    )

    decision = agent.run(SupportRequest(message="Is this accessory compatible?"))

    assert decision.action == Action.ESCALATE
    assert decision.reason == AgentEscalationReason.INSUFFICIENT_EVIDENCE
    assert len(llm.calls) == 1


def test_high_risk_message_escalates_deterministically():
    llm = FakeLLM([classification()])
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(
        SupportRequest(message="The charger caught fire and injured me.")
    )

    assert decision.reason == AgentEscalationReason.HIGH_RISK
    assert decision.proposed_reply is None


def test_account_security_always_escalates():
    llm = FakeLLM([classification("account_access_security")])
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(SupportRequest(message="I cannot log in to my account."))

    assert decision.reason == AgentEscalationReason.ACCOUNT_VERIFICATION_REQUIRED


def test_repeated_unresolved_history_escalates():
    llm = FakeLLM([classification()])
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(
        SupportRequest(
            message="This is still not resolved.",
            conversation_history=["I asked for help.", "Support replied twice."],
        )
    )

    assert decision.reason == AgentEscalationReason.REPEATED_UNRESOLVED


def test_malformed_classifier_output_fails_safe():
    llm = FakeLLM(
        [
            {
                "intent": "not_a_real_intent",
                "rationale": "",
            }
        ]
    )
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(SupportRequest(message="Please help with this product."))

    assert decision.reason == AgentEscalationReason.MODEL_FAILURE
    assert decision.intent.value == "other_unknown"


def test_api_failure_fails_safe():
    llm = FakeLLM([LLMResponseError("service unavailable")])
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(SupportRequest(message="Please help with this product."))

    assert decision.reason == AgentEscalationReason.MODEL_FAILURE
    assert "service unavailable" in decision.uncertainty


def test_unsupported_generated_claim_escalates():
    llm = FakeLLM(
        [
            classification(),
            {
                "reply": "We have issued your refund and it will arrive tomorrow.",
                "evidence_ids": ["case_1"],
            },
        ]
    )
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(SupportRequest(message="What is the general refund policy?"))

    assert decision.reason == AgentEscalationReason.UNSUPPORTED_CLAIM
    assert decision.proposed_reply is None


def test_malformed_reply_output_fails_safe():
    llm = FakeLLM(
        [
            classification(),
            {
                "reply": "",
                "evidence_ids": [],
            },
        ]
    )
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(SupportRequest(message="What sizes are available?"))

    assert decision.reason == AgentEscalationReason.MODEL_FAILURE
    assert decision.proposed_reply is None


def test_unknown_evidence_id_escalates():
    llm = FakeLLM(
        [
            classification(),
            {
                "reply": "Please review the public product details.",
                "evidence_ids": ["invented_case"],
            },
        ]
    )
    agent = SupportAgent(llm=llm, retriever=FakeRetriever(retrieval_result()))

    decision = agent.run(SupportRequest(message="What sizes are available?"))

    assert decision.reason == AgentEscalationReason.UNSUPPORTED_CLAIM


def test_blank_request_is_rejected():
    try:
        SupportRequest(message="   ")
    except ValueError:
        pass
    else:
        raise AssertionError("Blank requests must fail validation")


def test_unsupported_claim_detector_known_examples():
    policy = EscalationPolicy()
    assert policy.contains_unsupported_claim("We have processed your refund.")
    assert policy.contains_unsupported_claim("Your order will arrive tomorrow.")
    assert not policy.contains_unsupported_claim(
        "You can review general return guidance in the official help center."
    )
