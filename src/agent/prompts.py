"""Prompt construction for constrained classification and grounded drafting."""
import json

from src.data.intent_taxonomy import Intent
from src.knowledge.retriever import RetrievalResult

INTENT_SYSTEM_PROMPT = f"""You classify AmazonHelp customer messages.
Choose exactly one intent from: {", ".join(Intent.list_values())}.
Use other_unknown when the message is ambiguous or outside this taxonomy.
Return only the requested structured output. Do not invent confidence scores."""

REPLY_SYSTEM_PROMPT = """You draft concise offline AmazonHelp support replies.
Historical replies are examples of past handling, not proof of resolution or
current policy. Use only the supplied evidence. Never claim that a refund,
replacement, cancellation, delivery, account change, or investigation has
already happened. Never promise dates or outcomes. Never request passwords,
payment details, addresses, order numbers, or other sensitive data publicly.
When private follow-up is needed, ask the customer to use an official secure
support channel without requesting the private data in the public reply.
Return only the requested structured output and cite only supplied evidence IDs."""


def build_intent_prompt(message: str, history: list[str]) -> str:
    return json.dumps(
        {
            "customer_message": message,
            "conversation_history": history,
        },
        ensure_ascii=False,
    )


def build_reply_prompt(
    message: str,
    history: list[str],
    intent: Intent,
    retrieval: RetrievalResult,
) -> str:
    evidence = [
        {
            "source_id": record.pair_id,
            "historical_customer_issue": record.cleaned_customer_text,
            "historical_brand_reply": record.cleaned_brand_reply,
            "similarity": score,
        }
        for record, score in zip(retrieval.records, retrieval.scores)
    ]
    return json.dumps(
        {
            "customer_message": message,
            "conversation_history": history,
            "predicted_intent": intent.value,
            "historical_evidence": evidence,
        },
        ensure_ascii=False,
    )
