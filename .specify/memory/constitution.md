# realtime-fraud-detection-ml Constitution

## Core Principles

### I. Training and serving compute the same features
One feature module; a test compares the streaming path with the batch path on the same events.

### II. The test set is touched once
Model choices (hyperparameters, threshold) are made on data the reported metrics never see.
The order is time: train, then validation, then test.

### III. Correct under restart
Per-card state and the predictions survive a restart of the scoring job: no velocity history
lost, no prediction written twice. Shown by a test, not asserted.

### IV. Watched in production
Live precision, recall and input drift are measured against written thresholds, with an exit
code and a webhook, so a degraded model is noticed without opening the dashboard.

### V. Honest limits
The README and the model card state what the model misses and why.

## Constraints
Free: local Spark and Redpanda, GitHub Actions, Streamlit Community Cloud. Python 3.11,
Spark 3.5. Consumes the topic of realtime-fraud-streaming, including its schema registry format.

## Development Workflow
Spec Kit in `specs/`. Ruff and unit tests on every push; the pipeline check runs the real stream.

## Governance
A principle changes only with an ADR.

**Version**: 1.0.0 | **Ratified**: 2026-11-27 | **Last Amended**: 2026-11-27
