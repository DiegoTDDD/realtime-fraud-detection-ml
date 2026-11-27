# 0009. A monitor with written thresholds for quality and drift

**Status:** Accepted

## Context
The dashboard showed live precision and recall, but nothing acted on them: a degraded model was noticed only if someone happened to look.

## Decision
`scoring/monitor.py` reads the latest 2,000 predictions from Delta (at least 500 to judge, after the first five minutes of the stream, while velocity windows fill) and fails, with exit code 1 and a post to `ALERT_WEBHOOK_URL`, when:

| Check | Threshold |
|---|---|
| Precision | >= 0.90 |
| Recall | >= 0.90 |
| Recall on high_amount, card_testing | >= 0.90 |
| Recall on impossible_travel | >= 0.45 (the known weak pattern; 0.64 on the test period) |
| PSI of amount, distance from home, transactions in 5 min, time since last transaction | < 0.25 |

A quality check fails only when the evidence says the rate is below its floor: the upper end of the 95% Wilson interval must be under it. The first live run showed why: high_amount had 10 examples in the window and 8 were caught, 80% against a 90% floor, yet a model with 97% recall misses 2 of 10 about 3% of the time. A pattern with fewer than 10 examples in the window is skipped. The PSI reference (decile bins of each feature on the training period) is written by `train_model.py` to `training/reference_distribution.json`.

## Consequences
- The labels here come with each event. In a real deployment they arrive weeks later from chargebacks, so the quality checks would run on the window whose labels have arrived, while the drift checks run on the latest data at once.
- The pipeline check runs the monitor on the live stream after the restart; the rules are unit-tested with hand-made windows, including the webhook.
