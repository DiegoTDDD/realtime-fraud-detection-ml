"""The monitor's rules, on hand-made windows of predictions."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import monitor
import numpy as np
import pandas as pd


def window(n=1000, fraud_every=20, miss_every=0, false_alarms=0, amount=50.0):
    rng = np.random.default_rng(0)
    is_fraud = np.arange(n) % fraud_every == 0
    pred = is_fraud.copy()
    if miss_every:
        pred[np.flatnonzero(is_fraud)[::miss_every]] = False
    pred[np.flatnonzero(~is_fraud)[:false_alarms]] = True
    return pd.DataFrame(
        {
            "is_fraud": is_fraud,
            "predicted_fraud": pred,
            "fraud_type": np.where(is_fraud, "high_amount", None),
            "amount": rng.normal(amount, 5, n),
        }
    )


REFERENCE = {"amount": {"edges": [45.0, 50.0, 55.0], "shares": [0.16, 0.34, 0.34, 0.16]}}


def test_a_healthy_window_passes():
    failures, summary = monitor.evaluate(window(), REFERENCE, monitor.Thresholds())
    assert failures == [] and summary["precision"] == 1.0 and summary["recall"] == 1.0


def test_low_recall_and_low_pattern_recall_fail():
    failures, _ = monitor.evaluate(window(miss_every=4), REFERENCE, monitor.Thresholds())
    assert any(f.startswith("recall 0.") for f in failures)
    assert any("recall on high_amount" in f for f in failures)


def test_false_alarms_fail_precision():
    failures, _ = monitor.evaluate(window(false_alarms=20), REFERENCE, monitor.Thresholds())
    assert failures == ["precision 0.714 < 0.9"]


def test_a_shifted_feature_is_named_as_drift():
    failures, summary = monitor.evaluate(window(amount=70.0), REFERENCE, monitor.Thresholds())
    assert summary["psi"]["amount"] > 1
    assert failures == [f"drift in amount: PSI {summary['psi']['amount']:.3f} >= 0.25"]


def test_too_few_rows_or_examples_are_not_judged():
    failures, summary = monitor.evaluate(window(n=100), REFERENCE, monitor.Thresholds())
    assert failures == [] and "need 500" in summary["note"]
    sparse = window(fraud_every=200)  # 5 examples of the pattern, under the minimum of 10
    _, summary = monitor.evaluate(sparse, REFERENCE, monitor.Thresholds())
    assert summary["pattern_recall"] == {}


def test_psi_is_zero_for_the_training_distribution():
    values = np.repeat([40.0, 47.0, 52.0, 60.0], [16, 34, 34, 16])
    assert abs(monitor.psi(values, [45.0, 50.0, 55.0], [0.16, 0.34, 0.34, 0.16])) < 1e-9


def test_failures_are_posted_to_the_webhook():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    monitor.notify(f"http://127.0.0.1:{server.server_port}", ["recall 0.5 < 0.9"], {"rows": 1})
    server.server_close()
    assert "recall 0.5 < 0.9" in received[0]["text"]
