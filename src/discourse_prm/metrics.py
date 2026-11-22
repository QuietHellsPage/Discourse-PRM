"""
Module that contains method for additional metrics calculation
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """
    Method that computes extra training metrics

    Args:
      prediction (EvalPrediction): Model's prediction

    Returns:
      dict[str, float]: Resulting metrics
    """
    labels = prediction.label_ids
    predicted = np.asarray(prediction.predictions).argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "f1_macro": f1_score(
            labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0
        ),
    }
