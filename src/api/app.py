"""Local draft-only web interface for the support agent."""
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.agent.llm import OpenAIStructuredLLM
from src.agent.models import AgentDecision, AgentEscalationReason, SupportRequest
from src.agent.pipeline import SupportAgent
from src.config import load_config

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Hiver support draft agent")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app = FastAPI(title="Hiver support draft agent")


class DraftRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[str] = Field(default_factory=list, max_length=20)


class PublicGuidance(BaseModel):
    reply: str = Field(min_length=1, max_length=700)


PUBLIC_GUIDANCE_PROMPT = """You are AmazonHelp answering in a public chat.
Be specific and useful. You cannot see the customer's account, so do not claim
that a refund, replacement, delivery, cancellation, or investigation has
already happened. Do not invent dates or order status. Explain the normal
next step, and say a specialist must confirm anything account-specific.
Return only the structured reply."""


def public_guidance(agent: SupportAgent, request: DraftRequest, decision: AgentDecision) -> str | None:
    """Give a useful chat reply when policy blocks automatic handling."""
    if decision.proposed_reply:
        return decision.proposed_reply
    if decision.reason in {
        AgentEscalationReason.HIGH_RISK,
        AgentEscalationReason.MODEL_FAILURE,
    }:
        return None
    guidance = agent.llm.generate(
        system_prompt=PUBLIC_GUIDANCE_PROMPT,
        user_prompt=request.message,
        output_model=PublicGuidance,
    )
    return guidance.reply


@lru_cache(maxsize=1)
def get_agent() -> SupportAgent:
    config = load_config()
    if not config.agent.api_key or config.agent.api_key == "your_api_key_here":
        raise RuntimeError("OPENAI_API_KEY is required to draft replies.")
    from src.knowledge.retriever import FAISSRetriever

    retriever = FAISSRetriever(
        relevance_threshold=config.retrieval.relevance_threshold
    ).load(config.retrieval.faiss_index_path, config.retrieval.metadata_path)
    return SupportAgent(
        llm=OpenAIStructuredLLM(config.agent),
        retriever=retriever,
        top_k=config.retrieval.top_k,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/config.js")
def config_js() -> FileResponse:
    return FileResponse(WEB_DIR / "config.js", media_type="application/javascript")


@app.post("/api/draft")
def draft(request: DraftRequest) -> dict:
    try:
        agent = get_agent()
        decision = agent.run(
            SupportRequest(
                message=request.message.strip(),
                conversation_history=request.history,
            )
        )
        reply = public_guidance(agent, request, decision)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The agent could not complete this draft.") from exc
    return {
        "intent": decision.intent.value,
        "action": decision.action.value,
        "reason": decision.reason.value,
        "reason_detail": decision.reason_detail,
        "evidence_ids": decision.evidence_ids,
        "reply": reply,
        "uncertainty": decision.uncertainty,
    }
