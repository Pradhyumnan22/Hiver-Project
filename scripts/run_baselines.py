"""Script to evaluate baseline models on the held-out golden evaluation set and save results."""
import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.intent_taxonomy import Action, GoldenEvaluationExample, Intent
from src.evaluation.baselines import MajorityClassBaseline, SimpleRetrievalReplyBaseline, TfidfLogisticRegressionBaseline
from src.evaluation.metrics import calculate_classification_metrics, calculate_escalation_metrics
from src.utils.logging import setup_logger

logger = setup_logger("run_baselines")

DEV_PAIRS_PATH = Path("data/processed/dev_pairs.json")
GOLDEN_SET_PATH = Path("data/processed/golden_eval_set.json")
RESULTS_PATH = Path("reports/results.json")
RETRIEVAL_RELEVANCE_THRESHOLD = 0.35


def run_baselines():
    logger.info("Loading development pairs and golden evaluation benchmark...")
    with open(DEV_PAIRS_PATH, "r", encoding="utf-8") as f:
        dev_pairs = json.load(f)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    golden_examples = [GoldenEvaluationExample(**item) for item in golden_raw]

    dev_author_ids = {str(pair["customer_author_id"]) for pair in dev_pairs}
    dev_tweet_ids = {int(pair["customer_tweet_id"]) for pair in dev_pairs}
    golden_author_ids = {example.customer_author_id for example in golden_examples}
    golden_tweet_ids = {example.customer_tweet_id for example in golden_examples}
    if dev_author_ids & golden_author_ids or dev_tweet_ids & golden_tweet_ids:
        raise ValueError("Development/golden leakage detected; refusing to run baselines.")

    # Development labels are deterministic weak-supervision labels. Golden
    # labels were reviewed independently and are never used for model fitting.
    from scripts.build_golden_set import classify_candidate
    train_texts = []
    train_intents = []
    train_actions = []
    train_replies = []
    train_source_ids = []

    for p in dev_pairs:
        text = p["cleaned_customer_text"]
        brand_reply = p["cleaned_brand_text"]
        intent, action, _, _ = classify_candidate(text, brand_reply)
        train_texts.append(text)
        train_intents.append(intent.value)
        train_actions.append(action.value)
        train_replies.append(brand_reply)
        train_source_ids.append(p["pair_id"])

    eval_texts = [ex.cleaned_customer_message for ex in golden_examples]
    eval_true_intents = [ex.expected_intent.value for ex in golden_examples]
    eval_true_actions = [ex.expected_action.value for ex in golden_examples]
    all_intent_labels = [i.value for i in Intent]

    logger.info(f"Training set: {len(train_texts)} examples. Golden evaluation set: {len(eval_texts)} examples.")

    # 1. Majority Class Baseline
    logger.info("--- Evaluating Baseline 1: Majority Class ---")
    start_time = time.perf_counter()
    maj_model = MajorityClassBaseline().fit(train_texts, train_intents)
    maj_intent_preds = maj_model.predict(eval_texts)
    maj_latency_ms = round((time.perf_counter() - start_time) * 1000 / len(eval_texts), 3)

    # Escalation majority action
    maj_action_model = MajorityClassBaseline().fit(train_texts, train_actions)
    maj_action_preds = maj_action_model.predict(eval_texts)

    maj_classification_metrics = calculate_classification_metrics(eval_true_intents, maj_intent_preds, labels=all_intent_labels)
    maj_escalation_metrics = calculate_escalation_metrics(eval_true_actions, maj_action_preds)

    logger.info(f"Majority Class -> Accuracy: {maj_classification_metrics['accuracy']:.4f}, Macro F1: {maj_classification_metrics['macro_f1']:.4f}")

    # 2. TF-IDF + Logistic Regression Baseline
    logger.info("--- Evaluating Baseline 2: TF-IDF + Logistic Regression ---")
    start_time = time.perf_counter()
    tfidf_model = TfidfLogisticRegressionBaseline().fit(train_texts, train_intents)
    tfidf_intent_preds = tfidf_model.predict(eval_texts)
    tfidf_latency_ms = round((time.perf_counter() - start_time) * 1000 / len(eval_texts), 3)

    # Escalation logistic regression
    tfidf_action_model = TfidfLogisticRegressionBaseline().fit(train_texts, train_actions)
    tfidf_action_preds = tfidf_action_model.predict(eval_texts)

    tfidf_classification_metrics = calculate_classification_metrics(eval_true_intents, tfidf_intent_preds, labels=all_intent_labels)
    tfidf_escalation_metrics = calculate_escalation_metrics(eval_true_actions, tfidf_action_preds)

    logger.info(f"TF-IDF + LogReg -> Accuracy: {tfidf_classification_metrics['accuracy']:.4f}, Macro F1: {tfidf_classification_metrics['macro_f1']:.4f}")

    # 3. Simple Retrieval Reply Baseline
    logger.info("--- Evaluating Baseline 3: Simple Retrieval Reply ---")
    retriever = SimpleRetrievalReplyBaseline().fit(
        train_texts,
        train_replies,
        source_ids=train_source_ids,
    )
    retrieval_start = time.perf_counter()
    retrieval_outputs = [
        retriever.reply_with_source(example.cleaned_customer_message)
        for example in golden_examples
    ]
    retrieval_latency_ms = round(
        (time.perf_counter() - retrieval_start) * 1000 / len(golden_examples),
        3,
    )
    retrieval_scores = [score for _, score, _ in retrieval_outputs]
    coverage = sum(
        score >= RETRIEVAL_RELEVANCE_THRESHOLD for score in retrieval_scores
    ) / len(retrieval_scores)

    # Include one traceable example per intent rather than the first five rows,
    # which are grouped by intent in the finalized JSON.
    sample_replies = []
    seen_intents = set()
    for ex, (rep, score, source_id) in zip(golden_examples, retrieval_outputs):
        if ex.expected_intent in seen_intents:
            continue
        seen_intents.add(ex.expected_intent)
        sample_replies.append({
            "id": ex.id,
            "intent": ex.expected_intent.value,
            "query": ex.cleaned_customer_message,
            "retrieved_reply": rep,
            "similarity_score": score,
            "source_pair_id": source_id,
        })
        if len(seen_intents) == len(Intent):
            break

    # Assemble comprehensive results object
    results = {
        "benchmark_metadata": {
            "dataset": "AmazonHelp (Customer Support on Twitter)",
            "evaluation_set": "golden_eval_set.json",
            "eval_size": len(golden_examples),
            "train_size": len(train_texts),
            "training_label_source": "deterministic_weak_supervision_rules",
            "evaluation_label_source": "ai_manually_reviewed",
            "leakage_check": "zero author and tweet ID overlap",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "baselines": {
            "majority_class": {
                "classification": maj_classification_metrics,
                "escalation": maj_escalation_metrics,
                "avg_latency_ms_per_query": maj_latency_ms,
            },
            "tfidf_logistic_regression": {
                "classification": tfidf_classification_metrics,
                "escalation": tfidf_escalation_metrics,
                "avg_latency_ms_per_query": tfidf_latency_ms,
            },
            "simple_retrieval_reply": {
                "evaluated_queries": len(retrieval_scores),
                "mean_top1_similarity": round(statistics.mean(retrieval_scores), 4),
                "median_top1_similarity": round(statistics.median(retrieval_scores), 4),
                "relevance_threshold": RETRIEVAL_RELEVANCE_THRESHOLD,
                "coverage_at_threshold": round(coverage, 4),
                "avg_latency_ms_per_query": retrieval_latency_ms,
                "samples": sample_replies,
            },
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved complete baseline evaluation results to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_baselines()
