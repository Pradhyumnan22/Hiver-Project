"""Tests verifying baseline models, simple retrieval reply, and metric calculations."""
import pytest
from src.evaluation.baselines import MajorityClassBaseline, SimpleRetrievalReplyBaseline, TfidfLogisticRegressionBaseline
from src.evaluation.metrics import calculate_classification_metrics, calculate_escalation_metrics


def test_calculate_classification_metrics_known_values():
    # 2 classes: A, B.
    # True:  [A, A, B, B]
    # Pred:  [A, B, B, B]
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]

    # Acc = 3/4 = 0.75
    # Class A: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=0.6667
    # Class B: TP=2, FP=1, FN=0 -> Prec=2/3=0.6667, Rec=1.0, F1=0.80
    metrics = calculate_classification_metrics(y_true, y_pred, labels=["A", "B"])

    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["A"]["precision"] == 1.0
    assert metrics["per_class"]["A"]["recall"] == 0.5
    assert metrics["per_class"]["B"]["recall"] == 1.0
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 2


def test_calculate_escalation_metrics():
    y_true = ["escalate", "escalate", "auto_handle", "auto_handle"]
    y_pred = ["escalate", "auto_handle", "auto_handle", "escalate"]

    metrics = calculate_escalation_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert metrics["total_evaluated"] == 4
    assert metrics["escalation_rate"] == 0.5


def test_majority_class_baseline():
    train_texts = ["track package", "where is order", "delayed shipment", "damaged item"]
    train_labels = ["order_status_delivery", "order_status_delivery", "order_status_delivery", "damaged_defective_item"]

    model = MajorityClassBaseline().fit(train_texts, train_labels)
    preds = model.predict(["return my product", "hello"])

    assert len(preds) == 2
    assert preds[0] == "order_status_delivery"
    assert preds[1] == "order_status_delivery"


def test_tfidf_logistic_regression_baseline():
    train_texts = [
        "Where is my package? The tracking has not updated in days",
        "My order was supposed to be delivered yesterday",
        "I need a refund for my return",
        "Can I get a return label and my refund please?",
        "The item was shattered and damaged upon arrival",
        "Screen is broken and cracked on the phone",
    ]
    train_labels = [
        "order_status_delivery",
        "order_status_delivery",
        "returns_refunds",
        "returns_refunds",
        "damaged_defective_item",
        "damaged_defective_item",
    ]

    model = TfidfLogisticRegressionBaseline().fit(train_texts, train_labels)
    preds = model.predict(["package tracking late", "need return label"])

    assert len(preds) == 2
    assert preds[0] == "order_status_delivery"
    assert preds[1] == "returns_refunds"


def test_simple_retrieval_reply_baseline():
    queries = [
        "Where is my order? It is delayed.",
        "How do I return my broken shoes?",
    ]
    replies = [
        "Please check your tracking link in Your Orders for live carrier updates.",
        "You can initiate a return label at our Returns Center.",
    ]

    retriever = SimpleRetrievalReplyBaseline().fit(queries, replies)
    reply, score = retriever.reply("order is late tracking")

    assert "Your Orders" in reply
    assert score > 0.0
