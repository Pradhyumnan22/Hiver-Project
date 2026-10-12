"""Comprehensive classification and evaluation metric calculations."""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def calculate_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate macro F1, accuracy, per-class precision/recall/F1, and confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have equal length.")

    if not labels:
        labels = sorted(list(set(y_true).union(set(y_pred))))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = {}
    for lbl in labels:
        if lbl in report:
            per_class[lbl] = {
                "precision": round(float(report[lbl]["precision"]), 4),
                "recall": round(float(report[lbl]["recall"]), 4),
                "f1_score": round(float(report[lbl]["f1-score"]), 4),
                "support": int(report[lbl]["support"]),
            }
        else:
            per_class[lbl] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def calculate_escalation_metrics(
    y_true_actions: List[str],
    y_pred_actions: List[str],
) -> Dict[str, Any]:
    """Calculate accuracy, escalation rate, and binary metrics for escalation decisions."""
    if len(y_true_actions) != len(y_pred_actions):
        raise ValueError("y_true_actions and y_pred_actions must have equal length.")

    acc = float(accuracy_score(y_true_actions, y_pred_actions))
    # Binary metrics treating 'escalate' as positive class
    prec = float(precision_score(y_true_actions, y_pred_actions, pos_label="escalate", zero_division=0))
    rec = float(recall_score(y_true_actions, y_pred_actions, pos_label="escalate", zero_division=0))
    f1 = float(f1_score(y_true_actions, y_pred_actions, pos_label="escalate", zero_division=0))

    total = len(y_pred_actions)
    escalate_count = sum(1 for a in y_pred_actions if a == "escalate")
    escalation_rate = float(escalate_count / total) if total > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "escalation_f1": round(f1, 4),
        "escalation_precision": round(prec, 4),
        "escalation_recall": round(rec, 4),
        "escalation_rate": round(escalation_rate, 4),
        "total_evaluated": total,
    }
