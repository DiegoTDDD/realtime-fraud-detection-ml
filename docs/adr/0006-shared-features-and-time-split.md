# 0006. One feature module for training and serving; time-based split

**Status:** Accepted; the 80/20 split is replaced by 60/20/20 in [ADR 0007](0007-validation-split-for-the-threshold.md). Replaces separate feature code in `train_model.py` and `score_stream.py`.

## Context
Training and the scoring job each had their own copy of the feature code. The copies had drifted: training measured time in whole seconds, while serving used fractional seconds. A transaction 300.4 s after the previous one was inside the 5-minute window during training but outside it during serving. In addition, the home-city map was inferred from all rows, including the test period, so the test set leaked into a feature.

## Decision
- `training/features.py` holds every feature function. Training uses the batch functions, and the scoring job uses `featurize_batch` with `CardHistory`. Both use whole seconds.
- `tests/test_features.py` feeds the same transactions through both paths, in micro-batches of 1, 37 and 500 rows, and requires identical features.
- Data is split by time (earliest 80% to train), and home cities are inferred from the training rows only.

## Alternatives considered
- **A feature store (Feast).** The right tool across many models and teams, but heavy for five features.
- **Keep two copies and compare them in a test.** Still two places to change.

## Consequences
- Training-serving skew is now caught by a unit test on every push.
- With the leak removed, the metrics did not change: on this data, the home city learned from the first 80% of rows is the same as the one learned from all rows for every one of the 200 cards. The fix matters for data where that is not true.
