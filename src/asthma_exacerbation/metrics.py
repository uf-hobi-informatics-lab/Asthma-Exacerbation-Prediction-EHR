from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_j_statistic_metrics(
    true_labels,
    predictions,
    possible_labels=(0, 1),
    average: str = "weighted",
    probabilities=None,
):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average=average)
    precision = precision_score(true_labels, predictions, average=average, zero_division=1)
    recall = recall_score(true_labels, predictions, average=average, zero_division=1)
    cm = confusion_matrix(true_labels, predictions, labels=list(possible_labels))

    auroc = 0.0
    threshold = None
    adjusted_predictions = predictions

    try:
        positive_scores = probabilities[:, 1]
        auroc = roc_auc_score(true_labels, positive_scores)
        fpr, tpr, thresholds = roc_curve(
            true_labels,
            positive_scores,
            pos_label=possible_labels[1],
        )
        threshold = float(thresholds[np.argmax(tpr - fpr)])
        adjusted_predictions = (positive_scores >= threshold).astype(int)
        accuracy = accuracy_score(true_labels, adjusted_predictions)
        f1 = f1_score(true_labels, adjusted_predictions, average=average)
        precision = precision_score(
            true_labels,
            adjusted_predictions,
            average=average,
            zero_division=1,
        )
        recall = recall_score(true_labels, adjusted_predictions, average=average, zero_division=1)
        cm = confusion_matrix(true_labels, adjusted_predictions, labels=list(possible_labels))
    except (TypeError, ValueError, IndexError):
        pass

    true_positive_class = true_labels == possible_labels[1]
    predicted_positive_class = adjusted_predictions == possible_labels[1]

    tp = np.sum(true_positive_class & predicted_positive_class)
    tn = np.sum(~true_positive_class & ~predicted_positive_class)
    fp = np.sum(~true_positive_class & predicted_positive_class)
    fn = np.sum(true_positive_class & ~predicted_positive_class)

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": round(float(accuracy), 6),
        "f1": round(float(f1), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "auroc": round(float(auroc), 6),
        "sensitivity": round(float(sensitivity), 6),
        "specificity": round(float(specificity), 6),
        "confusion_matrix": cm.tolist(),
        "j_stat_threshold": threshold,
    }
