"""Splitting, threshold selection and per-pattern metrics for the fraud model.

Kept apart from train_model.py so each rule can be unit-tested on small,
hand-made inputs (tests/test_evaluation.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

FRAUD_TYPES = ["high_amount", "card_testing", "impossible_travel"]
MIN_PRECISION = 0.95  # alert budget: at most 1 false alarm in 20 alerts
N_CANDIDATES = 500  # thresholds tried, at evenly spaced quantiles of the scores


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float
    macro_recall: float  # mean recall over the fraud types present
    met_precision_floor: bool  # False: no threshold reached MIN_PRECISION, best F1 used


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.8, time_col: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest train_frac of rows for training, the rest for testing.

    A random split would leak through the velocity features, which depend on
    neighbouring transactions in time; a time split also matches how the model
    is used (trained on the past, scoring the future).
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    ordered = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * train_frac)
    return ordered.iloc[:cut], ordered.iloc[cut:]


def recall_by_fraud_type(
    fraud_type: pd.Series | Sequence[str | None],
    y_pred: np.ndarray | Sequence[int],
    fraud_types: Sequence[str] = FRAUD_TYPES,
) -> dict[str, float | None]:
    """Share of each fraud pattern's transactions that were flagged (None if absent)."""
    types = np.asarray(fraud_type, dtype=object)
    pred = np.asarray(y_pred)
    result: dict[str, float | None] = {}
    for name in fraud_types:
        mask = types == name
        result[name] = float(pred[mask].mean()) if mask.any() else None
    return result


def select_threshold(
    y_true: np.ndarray | Sequence[int],
    y_scores: np.ndarray | Sequence[float],
    fraud_type: pd.Series | Sequence[str | None],
    min_precision: float = MIN_PRECISION,
    n_candidates: int = N_CANDIDATES,
) -> ThresholdChoice:
    """Pick the threshold that maximizes mean recall across fraud types under a precision floor.

    Maximizing aggregate F1 instead lets the most frequent pattern
    (card_testing, ~83% of fraud here) dominate and can quietly drop a rare
    one. If no threshold reaches min_precision, falls back to the best-F1
    threshold and says so in the result.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)
    types = np.asarray(fraud_type, dtype=object)
    masks = [types == name for name in FRAUD_TYPES if (types == name).any()]

    best: ThresholdChoice | None = None
    for t in np.unique(np.quantile(y_scores, np.linspace(0, 1, n_candidates))):
        y_pred = (y_scores >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision < min_precision:
            continue
        macro = float(np.mean([y_pred[m].mean() for m in masks])) if masks else 0.0
        if best is None or macro > best.macro_recall:
            best = ThresholdChoice(
                threshold=float(t),
                precision=float(precision),
                recall=float(recall_score(y_true, y_pred)),
                macro_recall=macro,
                met_precision_floor=True,
            )
    if best is not None:
        return best

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    t = float(thresholds[int(np.argmax(f1[:-1]))])
    y_pred = (y_scores >= t).astype(int)
    return ThresholdChoice(
        threshold=t,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred)),
        macro_recall=float(np.mean([y_pred[m].mean() for m in masks])) if masks else 0.0,
        met_precision_floor=False,
    )
