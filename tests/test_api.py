"""Tests for the local draft web API."""
from fastapi.testclient import TestClient

from src.agent.models import AgentDecision, AgentEscalationReason
from src.api.app import app, get_agent
from src.data.intent_taxonomy import Action, Intent


class StubLLM:
    def generate(self, **kwargs):
        return kwargs["output_model"](
            reply="You can start the return from Your Orders. A specialist must confirm a missing-box exception."
        )


class StubAgent:
    llm = StubLLM()

    def run(self, request):
        assert request.message == "Where is the product guide?"
        return AgentDecision(
            action=Action.ESCALATE,
            reason=AgentEscalationReason.INSUFFICIENT_EVIDENCE,
            reason_detail="No historical evidence met the calibrated relevance threshold.",
            intent=Intent.PRODUCT_INQUIRY_AVAILABILITY,
            evidence_ids=[],
            proposed_reply=None,
        )


def test_draft_page_and_api(monkeypatch):
    monkeypatch.setattr("src.api.app.get_agent", lambda: StubAgent())
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "Hiver Desk" in page.text
    assert "offline draft" in page.text

    response = client.post(
        "/api/draft",
        json={"message": "Where is the product guide?", "history": ["Earlier question"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "escalate"
    assert body["intent"] == "product_inquiry_availability"
    assert body["reply"].startswith("You can start the return")
