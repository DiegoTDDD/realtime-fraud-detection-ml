"""Chronological split, per-pattern recall and threshold selection."""

import numpy as np
import pandas as pd
import pytest
from conftest import tx
from evaluation import chronological_split, recall_by_fraud_type, select_threshold


def scored(groups):
    """Build (y_true, scores, fraud_type) from [(fraud_type or None, score, count), ...]."""
    y, s, t = [], [], []
    for fraud_type, score, count in groups:
        y += [int(fraud_type is not None)] * count
        s += [score] * count
        t += [fraud_type] * count
    return np.array(y), np.array(s), t


def test_chronological_split_keeps_the_latest_rows_for_testing():
    df = pd.DataFrame([tx("A", s) for s in (40, 10, 30, 20, 50)])
    train, test = chronological_split(df, 0.6)
    assert list(train["transaction_id"]) == ["A-10", "A-20", "A-30"]
    assert list(test["transaction_id"]) == ["A-40", "A-50"]


@pytest.mark.parametrize("frac", [0, 1, 1.5])
def test_chronological_split_rejects_invalid_fractions(frac):
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame([tx("A", 1)]), frac)


def test_recall_by_fraud_type_reports_absent_patterns_as_none():
    result = recall_by_fraud_type(
        ["card_testing", "card_testing", "high_amount", None], [1, 0, 1, 1]
    )
    assert result == {"high_amount": 1.0, "card_testing": 0.5, "impossible_travel": None}


def test_threshold_favours_the_rare_pattern_that_best_f1_would_drop():
    # 100 card_testing at 0.9, 2 impossible_travel at 0.6, 3 legit at 0.62, 200 legit at 0.1.
    # Best F1 sits at 0.9 and misses impossible_travel; mean recall across patterns picks <= 0.6.
    y, s, t = scored(
        [
            ("card_testing", 0.9, 100),
            ("impossible_travel", 0.6, 2),
            (None, 0.62, 3),
            (None, 0.1, 200),
        ]
    )
    choice = select_threshold(y, s, t)
    assert choice.met_precision_floor
    assert 0.1 < choice.threshold <= 0.6
    assert choice.precision >= 0.95
    assert recall_by_fraud_type(t, (s >= choice.threshold).astype(int))["impossible_travel"] == 1.0


def test_threshold_respects_the_precision_floor_even_if_a_pattern_is_lost():
    # Catching impossible_travel would mean 20 legit alerts: precision far below 95%.
    y, s, t = scored(
        [
            ("card_testing", 0.9, 50),
            ("impossible_travel", 0.5, 2),
            (None, 0.55, 20),
            (None, 0.1, 200),
        ]
    )
    choice = select_threshold(y, s, t)
    assert choice.met_precision_floor and choice.precision >= 0.95
    assert choice.threshold > 0.55


def test_falls_back_to_best_f1_when_the_floor_is_unreachable():
    y, s, t = scored([("card_testing", 0.8, 10), (None, 0.8, 10), (None, 0.1, 50)])
    choice = select_threshold(y, s, t)
    assert not choice.met_precision_floor
    assert choice.threshold == pytest.approx(0.8)
