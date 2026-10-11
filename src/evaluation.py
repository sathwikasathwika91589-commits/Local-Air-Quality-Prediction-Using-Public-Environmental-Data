"""Evaluation helpers for chronological holdout predictions."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def regression_metrics(actual, predicted):
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}
