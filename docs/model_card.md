# Model card: real-time card fraud classifier

| | |
|---|---|
| **Model** | XGBoost binary classifier (200 trees, depth 5, learning rate 0.1) |
| **Version** | 1.1 (training code in `training/`, metrics in `training/metrics.json`) |
| **Output** | Fraud probability per transaction, plus an alert flag at threshold 0.5625 |
| **Serving** | Spark Structured Streaming `foreachBatch` over the `transactions` Kafka topic (`scoring/score_stream.py`) |
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

The data is split by time: the earliest 120,000 rows train the model and the latest 30,000 rows test it. A random split would leak information through the velocity features. Class imbalance is handled with `scale_pos_weight` = 12.17 instead of resampling.

| Metric (test set, threshold 0.5625) | Value |
|---|---|
| PR-AUC | 0.9925 |
| Precision | 95.75% |
| Recall | 96.55% |
| Mean recall across the three patterns | 86.98% |

| Fraud pattern | Test examples | Recall |
|---|---|---|
| card_testing | 1,772 | 100% |
| high_amount | 185 | 96.2% |
| impossible_travel | 190 | 64.7% |

**Threshold policy.** The threshold maximizes the mean recall across the three patterns while keeping precision at 95% or more, which caps false alarms at 1 in 20 alerts. The best-F1 threshold would instead be driven by card testing, the most frequent pattern (see [ADR 0004](adr/0004-threshold-policy.md)).

**Live monitoring.** The producer injects ground-truth labels, so the dashboard shows live precision and recall per pattern next to these offline numbers.

## Limitations

- **Synthetic data.** The fraud patterns are simple and fixed, which explains the near-perfect card-testing recall. Real fraud is adaptive, and these numbers do not carry over to real transactions.
- **impossible_travel recall is 64.7%.** About 13.5% of legitimate transactions also happen away from the home city, and a fraudulent charge looks like an ordinary trip once its distance is accounted for. The test set has only 190 examples, so this figure has a margin of about ±7 points. Three feature designs were compared (see [ADR 0002](adr/0002-distance-feature.md)).
- **Home cities are fixed at training time.** A card that is new or that moves is treated as being at home (distance 0) until the model is retrained.
- **Process-local velocity state.** Velocity features are correct only while the scoring job runs as one process ([ADR 0003](adr/0003-process-local-velocity-state.md)). A restart resets them.
- **No drift monitoring or retraining schedule.** The live precision and recall are shown on the dashboard but do not trigger anything.

## Risks

- **False positives** block legitimate purchases. The 95% precision floor bounds them on test data, but not on a shifted live distribution.
- **False negatives** concentrate in impossible_travel. A fraudster who stays in the card's home city and avoids extreme amounts and bursts is not modelled at all.
- **Location as a signal** can disadvantage people who travel often or live near city borders. A real deployment would need fairness checks by customer segment, and those segments do not exist in this data.
