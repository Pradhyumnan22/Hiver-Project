"""Run the live agent and optional LLM judge on the fixed golden benchmark."""
import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.llm import LLMResponseError, OpenAIStructuredLLM
from src.agent.models import SupportRequest
from src.agent.pipeline import SupportAgent
from src.config import load_config
from src.data.intent_taxonomy import GoldenEvaluationExample, Intent
from src.evaluation.agreement import calculate_judge_human_agreement
from src.evaluation.judge import JUDGE_CRITERIA, judge_reply
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_escalation_metrics,
)
from src.knowledge.retriever import FAISSRetriever, RetrievalResult
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_agent")
OUTPUT_PATH = Path("reports/agent_results.json")
HUMAN_RATINGS_PATH = Path("data/evaluation/human_ratings.csv")


class FixedRetriever:
    """Return precomputed evidence so concurrent LLM calls remain isolated."""

    def __init__(self, result: RetrievalResult):
        self.result = result

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        return self.result


def _load_completed_human_ratings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    completed = []
    for row in rows:
        if all(row.get(criterion, "").strip() for criterion in JUDGE_CRITERIA):
            completed.append(row)
    return completed


def run_evaluation(
    limit: int | None,
    skip_judge: bool,
    workers: int = 8,
    offset: int = 0,
) -> dict:
    config = load_config()
    if not config.agent.api_key or config.agent.api_key == "your_api_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is required for live agent and judge evaluation."
        )

    with open(config.data.golden_set_path, "r", encoding="utf-8") as f:
        golden = [GoldenEvaluationExample(**row) for row in json.load(f)]
    golden = golden[offset:]
    if limit is not None:
        golden = golden[:limit]

    retriever = FAISSRetriever().load(
        config.retrieval.faiss_index_path,
        config.retrieval.metadata_path,
    )
    agent_llm = OpenAIStructuredLLM(config.agent)

    judge_llm = None
    if not skip_judge:
        judge_settings = config.agent.model_copy(
            update={"llm_model": config.evaluation.judge_model}
        )
        judge_llm = OpenAIStructuredLLM(judge_settings)

    benchmark_start = time.perf_counter()
    retrieval_outputs = []
    for example in golden:
        retrieval_start = time.perf_counter()
        retrieval = retriever.retrieve(
            example.cleaned_customer_message,
            top_k=config.retrieval.top_k,
        )
        retrieval_outputs.append(
            (retrieval, (time.perf_counter() - retrieval_start) * 1000)
        )

    def evaluate_one(item):
        example, (retrieval, retrieval_latency_ms) = item
        agent = SupportAgent(
            llm=agent_llm,
            retriever=FixedRetriever(retrieval),
            top_k=config.retrieval.top_k,
        )
        start = time.perf_counter()
        decision = agent.run(
            SupportRequest(
                message=example.cleaned_customer_message,
                conversation_history=[],
            )
        )
        latency_ms = retrieval_latency_ms + (time.perf_counter() - start) * 1000
        evidence_by_id = {
            record.pair_id: {
                "source_id": record.pair_id,
                "customer_issue": record.cleaned_customer_text,
                "historical_reply": record.cleaned_brand_reply,
                "intent": record.intent,
            }
            for record in retrieval.records
        }
        evidence = [
            evidence_by_id[source_id]
            for source_id in decision.evidence_ids
            if source_id in evidence_by_id
        ]
        row = {
            "id": example.id,
            "expected_intent": example.expected_intent.value,
            "predicted_intent": decision.intent.value,
            "expected_action": example.expected_action.value,
            "predicted_action": decision.action.value,
            "reason": decision.reason.value,
            "reason_detail": decision.reason_detail,
            "uncertainty": decision.uncertainty,
            "reply": decision.proposed_reply,
            "evidence_ids": decision.evidence_ids,
            "retrieval_best_score": retrieval.best_score,
            "latency_ms": round(latency_ms, 3),
        }

        judge_row = None
        if judge_llm and decision.proposed_reply:
            try:
                rating = judge_reply(
                    judge_llm,
                    customer_message=example.cleaned_customer_message,
                    predicted_intent=decision.intent.value,
                    reply=decision.proposed_reply,
                    evidence=evidence,
                )
                row["judge"] = rating.model_dump()
                judge_row = {
                    "id": example.id,
                    **{
                        criterion: getattr(rating, criterion).score
                        for criterion in JUDGE_CRITERIA
                    },
                }
            except LLMResponseError as exc:
                row["judge_error"] = str(exc)
        return row, judge_row

    inputs = list(zip(golden, retrieval_outputs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        evaluated = list(executor.map(evaluate_one, inputs))

    predictions = []
    judge_rows = []
    for index, (row, judge_row) in enumerate(evaluated, start=1):
        predictions.append(row)
        if judge_row:
            judge_rows.append(judge_row)
        logger.info(f"Evaluated {index}/{len(golden)}: {row['id']}")
    latencies = [row["latency_ms"] for row in predictions]
    wall_time_seconds = time.perf_counter() - benchmark_start

    intent_metrics = calculate_classification_metrics(
        [row["expected_intent"] for row in predictions],
        [row["predicted_intent"] for row in predictions],
        labels=Intent.list_values(),
    )
    escalation_metrics = calculate_escalation_metrics(
        [row["expected_action"] for row in predictions],
        [row["predicted_action"] for row in predictions],
    )
    retrieval_covered = [
        bool(row["evidence_ids"]) for row in predictions
    ]
    retrieval_matches = [
        evidence_by_id_match
        for row in predictions
        if row["evidence_ids"]
        for evidence_by_id_match in [
            next(
                (
                    record.intent == row["expected_intent"]
                    for record in retriever.records
                    if record.pair_id == row["evidence_ids"][0]
                ),
                False,
            )
        ]
    ]

    judge_summary = None
    if judge_rows:
        judge_summary = {
            "rated_responses": len(judge_rows),
            "mean_scores": {
                criterion: round(
                    statistics.mean(row[criterion] for row in judge_rows),
                    4,
                )
                for criterion in JUDGE_CRITERIA
            },
        }

    human_rows = _load_completed_human_ratings(HUMAN_RATINGS_PATH)
    agreement = None
    if human_rows and judge_rows:
        agreement = calculate_judge_human_agreement(human_rows, judge_rows)

    result = {
        "metadata": {
            "evaluation_set": config.data.golden_set_path,
            "evaluated_examples": len(predictions),
            "agent_model": config.agent.llm_model,
            "judge_model": None if skip_judge else config.evaluation.judge_model,
            "temperature": config.agent.temperature,
            "retrieval_threshold": retriever.relevance_threshold,
            "workers": workers,
            "wall_time_seconds": round(wall_time_seconds, 3),
            "cost_usd": None,
            "cost_note": (
                "Token usage is recorded; USD cost is omitted because provider "
                "and model pricing are not assumed."
            ),
        },
        "classification": intent_metrics,
        "escalation": escalation_metrics,
        "retrieval": {
            "coverage": round(sum(retrieval_covered) / len(predictions), 4),
            "top1_same_intent_proxy_precision": (
                round(sum(retrieval_matches) / len(retrieval_matches), 4)
                if retrieval_matches
                else None
            ),
        },
        "reply_quality": judge_summary,
        "judge_human_agreement": agreement,
        "latency": {
            "mean_ms": round(statistics.mean(latencies), 3),
            "median_ms": round(statistics.median(latencies), 3),
            "max_ms": round(max(latencies), 3),
        },
        "usage": {
            "agent": agent_llm.usage,
            "judge": judge_llm.usage if judge_llm else None,
        },
        "predictions": predictions,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_evaluation(
        limit=args.limit,
        skip_judge=args.skip_judge,
        workers=args.workers,
        offset=args.offset,
    )
