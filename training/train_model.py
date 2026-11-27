"""Train the fraud classifier.

Pipeline:
  1. Load the generated training data (data/training_raw.parquet).
  2. Split chronologically: earliest 60% for training, the next 20% for
     validation, the latest 20% for testing (ADR 0007).
  3. Infer each card's home city from the training rows only, then build the
     features (training/features.py, shared with the scoring job).
  4. Train XGBoost with scale_pos_weight for the ~7.5% fraud rate instead of
     resampling (ADR 0005).
  5. Choose the alert threshold on the validation period: the one that
     maximizes mean recall across patterns under a 95% precision floor
     (ADR 0004).
  6. Report PR-AUC, precision, recall and recall per pattern on the test
     period, which no choice has seen.
  7. Save the training distribution of the monitored features for drift
     checks (scoring/monitor.py, ADR 0009).

Outputs: training/model.joblib (gitignored), training/card_home_city.json,
training/metrics.json, training/reference_distribution.json and
docs/screenshots/pr_curve.png.

Usage:
    python training/train_model.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed; also keeps CI and Windows runs headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (  # noqa: E402
    MIN_PRECISION,
    recall_by_fraud_type,
    select_threshold,
    three_way_split,
)
from features import build_features, get_home_city_map  # noqa: E402

log = logging.getLogger(__name__)

DATA_PATH = Path("data/training_raw.parquet")
MODEL_PATH = Path("training/model.joblib")
CARD_HOME_CITY_PATH = Path("training/card_home_city.json")
METRICS_PATH = Path("training/metrics.json")
PR_CURVE_PATH = Path("docs/screenshots/pr_curve.png")
REFERENCE_PATH = Path("training/reference_distribution.json")
MONITORED_FEATURES = ["amount", "distance_km_from_home", "tx_count_last_5min", "time_since_last_tx"]

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eval_metric": "aucpr",
    "random_state": 42,
}


def prepare(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    """Split by time, infer home cities from the training part, build features for every part."""
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="ISO8601")
    train_raw, _, _ = three_way_split(raw)
    home_city = get_home_city_map(train_raw)  # never looks at the validation or test periods
    df, feature_cols = build_features(raw, home_city)
    train_df, val_df, test_df = three_way_split(df)
    return train_df, val_df, test_df, feature_cols, home_city


def reference_distribution(train_df: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Decile edges of each monitored feature on the training rows, and the share in each bin.

    Velocity features have many equal values (most cards are not bursting), so the edges are
    de-duplicated and the shares measured on the bins that remain.
    """
    out = {}
    for name in MONITORED_FEATURES:
        values = train_df[name].astype(float).to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        out[name] = {
            "edges": [float(e) for e in inner],
            "shares": [float(c) / len(values) for c in counts],
        }
    return out


def train(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[XGBClassifier, float]:
    """Fit XGBoost with class weights; return the model and the weight used."""
    y = train_df["is_fraud"].astype(int)
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    model.fit(train_df[feature_cols], y)
    return model, scale_pos_weight


def save_pr_curve(
    y_true: pd.Series, y_scores: Any, pr_auc: float, point: tuple[float, float], threshold: float
) -> None:
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label=f"PR-AUC = {pr_auc:.3f}")
    plt.scatter(
        [point[0]], [point[1]], color="red", zorder=5, label=f"Chosen threshold ({threshold:.3f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Fraud Classifier")
    plt.legend()
    plt.tight_layout()
    PR_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PR_CURVE_PATH, dpi=150)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("Loading %s...", DATA_PATH)
    train_df, val_df, test_df, feature_cols, home_city = prepare(pd.read_parquet(DATA_PATH))
    y_val = val_df["is_fraud"].astype(int)
    y_test = test_df["is_fraud"].astype(int)
    log.info("Train: %s rows (%s fraud)", f"{len(train_df):,}", f"{train_df['is_fraud'].sum():,}")
    log.info("Val:   %s rows (%s fraud)", f"{len(val_df):,}", f"{y_val.sum():,}")
    log.info("Test:  %s rows (%s fraud)", f"{len(test_df):,}", f"{y_test.sum():,}")

    model, scale_pos_weight = train(train_df, feature_cols)
    log.info("Trained XGBoost, scale_pos_weight = %.2f", scale_pos_weight)

    # The threshold is chosen on the validation period ...
    val_scores = model.predict_proba(val_df[feature_cols])[:, 1]
    choice = select_threshold(y_val, val_scores, val_df["fraud_type"])
    if not choice.met_precision_floor:
        log.warning(
            "No threshold reached precision %.2f; using the best-F1 threshold", MIN_PRECISION
        )
    # ... and everything reported comes from the test period, scored once.
    y_scores = model.predict_proba(test_df[feature_cols])[:, 1]
    pr_auc = float(average_precision_score(y_test, y_scores))
    y_pred = (y_scores >= choice.threshold).astype(int)
    recalls = recall_by_fraud_type(test_df["fraud_type"], y_pred)
    test_precision = float(precision_score(y_test, y_pred, zero_division=0))
    test_recall = float(recall_score(y_test, y_pred, zero_division=0))
    present = [v for v in recalls.values() if v is not None]
    test_macro = float(np.mean(present)) if present else 0.0

    log.info("PR-AUC: %.4f", pr_auc)
    log.info("Threshold (macro recall, precision >= %.2f): %.4f", MIN_PRECISION, choice.threshold)
    log.info("Validation precision %.4f recall %.4f", choice.precision, choice.recall)
    log.info("Test precision: %.4f  Recall: %.4f", test_precision, test_recall)
    for name, value in recalls.items():
        log.info("  recall %-18s %s", name, "n/a" if value is None else f"{value:.4f}")

    save_pr_curve(y_test, y_scores, pr_auc, (test_recall, test_precision), choice.threshold)
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "threshold": choice.threshold}, MODEL_PATH
    )
    CARD_HOME_CITY_PATH.write_text(json.dumps(home_city, indent=2), encoding="utf-8")
    REFERENCE_PATH.write_text(
        json.dumps(reference_distribution(train_df), indent=2), encoding="utf-8"
    )
    metrics = {
        "split": "chronological 60/20/20; threshold chosen on validation, metrics on test",
        "pr_auc": pr_auc,
        "threshold": choice.threshold,
        "precision": test_precision,
        "recall": test_recall,
        "macro_recall": test_macro,
        "validation_precision": choice.precision,
        "validation_recall": choice.recall,
        "recall_by_fraud_type": recalls,
        "test_count_by_fraud_type": {
            name: int((test_df["fraud_type"] == name).sum()) for name in recalls
        },
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "train_fraud_count": int(train_df["is_fraud"].sum()),
        "test_fraud_count": int(y_test.sum()),
        "scale_pos_weight": scale_pos_weight,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(
        "Saved %s, %s, %s, %s and %s",
        MODEL_PATH,
        CARD_HOME_CITY_PATH,
        METRICS_PATH,
        REFERENCE_PATH,
        PR_CURVE_PATH,
    )


if __name__ == "__main__":
    main()
