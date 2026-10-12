"""Script to evaluate baseline models on the held-out golden evaluation set and save results."""
import json
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


def run_baselines():
    logger.info("Loading development pairs and golden evaluation benchmark...")
    with open(DEV_PAIRS_PATH, "r", encoding="utf-8") as f:
        dev_pairs = json.load(f)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_raw = json.load(f)
    golden_examples = [GoldenEvaluationExample(**item) for item in golden_raw]

    # Prepare training data from dev set using annotation heuristic logic
    from scripts.build_golden_set import classify_candidate
    train_texts = []
    train_intents = []
    train_actions = []
    train_replies = []

    for p in dev_pairs:
        text = p["cleaned_customer_text"]
        brand_reply = p["cleaned_brand_text"]
        intent, action, _, _ = classify_candidate(text, brand_reply)
        train_texts.append(text)
        train_intents.append(intent.value)
        train_actions.append(action.value)
        train_replies.append(brand_reply)

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
    retriever = SimpleRetrievalReplyBaseline().fit(train_texts, train_replies)
    sample_replies = []
    for ex in golden_examples[:5]:
        rep, score = retriever.reply(ex.cleaned_customer_message)
        sample_replies.append({
            "id": ex.id,
            "query": ex.cleaned_customer_message,
            "retrieved_reply": rep,
            "similarity_score": score,
        })

    # Assemble comprehensive results object
    results = {
        "benchmark_metadata": {
            "dataset": "AmazonHelp (Customer Support on Twitter)",
            "evaluation_set": "golden_eval_set.json",
            "eval_size": len(golden_examples),
            "train_size": len(train_texts),
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
