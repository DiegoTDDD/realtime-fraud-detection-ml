"""Model monitor: live precision, recall, recall per fraud pattern and input drift.

Reads the latest predictions from the Delta table written by score_stream.py and checks them
against written thresholds (ADR 0009). Exits 1 and posts to ALERT_WEBHOOK_URL (if set) when a
check fails, so a degraded model is noticed without opening the dashboard.

  - precision and recall over the latest WINDOW predictions, against the labels the producer
    attaches to each event (a real deployment gets labels later, from chargebacks; the checks
    are the same, run on the window whose labels have arrived);
  - recall per fraud pattern, against a floor per pattern, skipped when a pattern has too few
    examples in the window to judge;
  - PSI of each monitored feature against its training distribution
    (training/reference_distribution.json).

The first VELOCITY_WINDOW_S of a stream are left out: velocity features need five minutes of
history before they look like the training data.

Run:  python scoring/monitor.py [--predictions data/predictions] [--window 2000]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from features import VELOCITY_WINDOW_S, add_stream_features  # noqa: E402

log = logging.getLogger(__name__)
REFERENCE_PATH = ROOT / "training" / "reference_distribution.json"
HOME_CITY_PATH = ROOT / "training" / "card_home_city.json"
PREDICTIONS_DIR = ROOT / "data" / "predictions"


@dataclass(frozen=True)
class Thresholds:
    min_precision: float = 0.90
    min_recall: float = 0.90
    pattern_recall: dict[str, float] = field(
        default_factory=lambda: {
            "high_amount": 0.90,
            "card_testing": 0.90,
            "impossible_travel": 0.45,
        }
    )
    min_pattern_examples: int = 10
    max_psi: float = 0.25
    min_rows: int = 500


def psi(values: np.ndarray, edges: list[float], expected: list[float], eps: float = 1e-4) -> float:
    """Population stability index of `values` over the training bins."""
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(expected))
    actual = np.clip(counts / max(len(values), 1), eps, None)
    base = np.clip(np.asarray(expected), eps, None)
    return float(np.sum((actual - base) * np.log(actual / base)))


def evaluate(df: pd.DataFrame, reference: dict, t: Thresholds) -> tuple[list[str], dict]:
    """Return (failures, summary) for a window of scored predictions with features."""
    summary: dict = {"rows": len(df)}
    if len(df) < t.min_rows:
        summary["note"] = f"only {len(df)} predictions; need {t.min_rows} to judge"
        return [], summary
    failures: list[str] = []
    y = df["is_fraud"].astype(bool).to_numpy()
    pred = df["predicted_fraud"].astype(bool).to_numpy()
    tp = int((y & pred).sum())
    precision = tp / pred.sum() if pred.sum() else math.nan
    recall = tp / y.sum() if y.sum() else math.nan
    summary.update(precision=precision, recall=recall, frauds=int(y.sum()), alerts=int(pred.sum()))
    if not math.isnan(precision) and precision < t.min_precision:
        failures.append(f"precision {precision:.3f} < {t.min_precision}")
    if not math.isnan(recall) and recall < t.min_recall:
        failures.append(f"recall {recall:.3f} < {t.min_recall}")
    summary["pattern_recall"] = {}
    for name, floor in t.pattern_recall.items():
        mask = (df["fraud_type"] == name).to_numpy()
        if mask.sum() < t.min_pattern_examples:
            continue
        r = float(pred[mask].mean())
        summary["pattern_recall"][name] = r
        if r < floor:
            failures.append(f"recall on {name} {r:.3f} < {floor}")
    summary["psi"] = {}
    for name, spec in reference.items():
        value = psi(df[name].astype(float).to_numpy(), spec["edges"], spec["shares"])
        summary["psi"][name] = value
        if value >= t.max_psi:
            failures.append(f"drift in {name}: PSI {value:.3f} >= {t.max_psi}")
    return failures, summary


def load_window(path: Path, window: int) -> pd.DataFrame:
    """The latest `window` predictions after the warm-up, with the drift features added."""
    from deltalake import DeltaTable

    df = DeltaTable(str(path)).to_pandas()
    df = df.sort_values("event_time").drop_duplicates("transaction_id", keep="last")
    start = df["event_time"].min()
    df = df[df["event_time"] >= start + pd.Timedelta(seconds=VELOCITY_WINDOW_S)].tail(window)
    home = json.loads(HOME_CITY_PATH.read_text(encoding="utf-8"))
    return add_stream_features(df, home, [])


def notify(url: str, failures: list[str], summary: dict) -> None:
    body = json.dumps(
        {"text": "Fraud model monitor failed:\n- " + "\n- ".join(failures), "summary": summary}
    )
    request = urllib.request.Request(url, body.encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10):
        pass


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", default=str(PREDICTIONS_DIR))
    parser.add_argument("--window", type=int, default=2000)
    args = parser.parse_args(argv)
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    failures, summary = evaluate(
        load_window(Path(args.predictions), args.window), reference, Thresholds()
    )
    log.info(json.dumps(summary, indent=2, default=float))
    if not failures:
        log.info("monitor: ok")
        return 0
    for failure in failures:
        log.error("monitor: %s", failure)
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if url:
        notify(url, failures, summary)
    return 1


if __name__ == "__main__":
    code = main()
    logging.shutdown()
    sys.stdout.flush()
    # deltalake's background threads can abort a normal interpreter exit ("terminate called
    # without an active exception"), which would turn a pass into exit code 134.
    os._exit(code)
