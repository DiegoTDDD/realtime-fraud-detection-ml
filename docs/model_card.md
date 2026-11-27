# Model card: real-time card fraud classifier

| | |
|---|---|
| **Model** | XGBoost binary classifier (200 trees, depth 5, learning rate 0.1) |
| **Version** | 1.2 (training code in `training/`, metrics in `training/metrics.json`) |
| **Output** | Fraud probability per transaction, plus an alert flag at threshold 0.578 |
| **Serving** | Spark Structured Streaming over the `transactions` Kafka topic, per-card state in Spark's state store, predictions in Delta (`scoring/score_stream.py`) |
| **Monitoring** | `scoring/monitor.py`: precision, recall, recall per pattern and feature drift (PSI) against written thresholds |
| **Owner** | Diego Alves |

## Intended use

Scoring card transactions from the [realtime-fraud-streaming](https://github.com/DiegoTDDD/realtime-fraud-streaming) producer as they arrive, to flag likely fraud for review. It is a portfolio system trained on simulated data. It is **not** fit to make decisions about real cardholders.

## Training data

- 150,000 synthetic transactions written by `training/generate_training_data.py`, which reuses the producer's simulation logic: 200 cards, 10 Brazilian cities, 8 merchant categories, about 4.2 hours of simulated time starting 2026-01-01, at the live producer's pace of about 10 transactions per second (so the velocity features match the stream the model scores).
- Three injected fraud patterns:
  - **high_amount:** R$5,000 to R$15,000 in the card's home city.
  - **card_testing:** bursts of 5 to 12 charges of R$0.50 to R$5.00 on one card.
  - **impossible_travel:** a charge in a city other than the card's home city.
- The data is fully reproducible: the random seed and the start date are fixed.
- Fraud makes up 7.5% of rows (11,261 of 150,000). The producer starts a fraud event 2% of the time, but each card-testing event adds a burst of extra fraudulent rows, so 83% of fraudulent rows are card testing.

## Features

All features are computed by `training/features.py`, which training and serving share. A unit test checks that the streaming path reproduces the training features exactly.

| Feature | Targets |
|---|---|
| `amount` | high_amount |
| `distance_km_from_home` | impossible_travel. Great-circle distance to the card's home city, which is inferred from the training period only |
| `tx_count_last_5min`, `time_since_last_tx` | card_testing |
| `hour_of_day`, one-hot `category` | general behaviour |

## Evaluation

The data is split by time into three consecutive periods: the earliest 90,000 rows train the model, the next 30,000 choose the threshold, and the latest 30,000 are scored once for the numbers below ([ADR 0007](adr/0007-validation-split-for-the-threshold.md)). A random split would leak information through the velocity features. Class imbalance is handled with `scale_pos_weight` = 12.13 instead of resampling.

| Metric (test period, threshold 0.578 chosen on validation) | Value |
|---|---|
| PR-AUC | 0.9918 |
| Precision | 96.60% (validation: 96.11%) |
| Recall | 96.60% (validation: 97.30%) |
| Mean recall across the three patterns | 87.17% |

| Fraud pattern | Test examples | Recall |
|---|---|---|
| card_testing | 1,772 | 100% |
| high_amount | 185 | 97.3% |
| impossible_travel | 190 | 64.2% |

**Threshold policy.** The threshold maximizes the mean recall across the three patterns while keeping precision at 95% or more, which caps false alarms at 1 in 20 alerts. The best-F1 threshold would instead be driven by card testing, the most frequent pattern (see [ADR 0004](adr/0004-threshold-policy.md)).

**Live monitoring.** The producer injects ground-truth labels, so the dashboard shows live precision and recall per pattern next to these offline numbers, and `scoring/monitor.py` fails with an alert when precision or recall drops below 90%, a pattern drops below its floor, or a feature's PSI against training reaches 0.25 ([ADR 0009](adr/0009-model-monitor.md)).

## Limitations

- **Synthetic data.** The fraud patterns are simple and fixed, which explains the near-perfect card-testing recall. Real fraud is adaptive, and these numbers do not carry over to real transactions.
- **impossible_travel recall is 64.2%.** About 13.5% of legitimate transactions also happen away from the home city, and a fraudulent charge looks like an ordinary trip once its distance is accounted for. The test set has only 190 examples, so this figure has a margin of about ±7 points. Three feature designs were compared (see [ADR 0002](adr/0002-distance-feature.md)).
- **Home cities are fixed at training time.** A card that is new or that moves is treated as being at home (distance 0) until the model is retrained.
- **Labels arrive with each event.** Here the producer supplies them; real labels come weeks later from chargebacks, so live precision and recall would lag the drift checks.
- **No retraining schedule.** The monitor raises the alarm; retraining is a manual run of `train_model.py`.
- **A card idle for 24 hours counts as new.** Its state is dropped; training applies the same cap ([ADR 0008](adr/0008-spark-state-and-idempotent-delta-sink.md)).

## Risks

- **False positives** block legitimate purchases. The 95% precision floor bounds them on validation data, but not on a shifted live distribution.
- **False negatives** concentrate in impossible_travel. A fraudster who stays in the card's home city and avoids extreme amounts and bursts is not modelled at all.
- **Location as a signal** can disadvantage people who travel often or live near city borders. A real deployment would need fairness checks by customer segment, and those segments do not exist in this data.
