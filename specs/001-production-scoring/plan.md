# Implementation Plan: Honest threshold, state that survives restarts, and model monitoring

**Branch**: `001-production-scoring` | **Date**: 2026-11-27 | **Spec**: [spec.md](spec.md)

## Summary

Split training data into train, validation and test by time and choose the threshold on
validation. Move per-card velocity state into Spark's state store with applyInPandasWithState,
write predictions to Delta with the batch id as the idempotent transaction version, accept the
schema registry wire format, and add a monitor with precision, recall, per-pattern recall and
PSI checks.

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**: PySpark 3.5.8 (applyInPandasWithState), delta-spark 3.3.2, deltalake
1.6.6, XGBoost 2.1, scikit-learn 1.5

**Storage**: Delta `data/predictions`, checkpoint with state in `data/_checkpoints/predictions`

**Testing**: pytest (no Spark) on every push; `pytest -m spark` in the pipeline check

**Constraints**: free; reads realtime-fraud-streaming's topic

## Constitution Check

| Principle | How |
|---|---|
| I. Same features | State function reuses the CardHistory window logic; Spark test compares with batch features. |
| II. Test touched once | 60/20/20 by time; threshold on validation. |
| III. Correct under restart | State store + Delta idempotent writes; restart test and CI kill. |
| IV. Watched | monitor.py, thresholds, PSI reference from training, webhook. |
| V. Honest limits | README and model card updated with new numbers and remaining limits. |

## Project Structure

```text
training/evaluation.py     # three-way chronological split
training/train_model.py    # threshold on validation; reference distribution for PSI
training/features.py       # velocity_for_card(): the state function's core
scoring/score_stream.py    # wire format, applyInPandasWithState, Delta idempotent sink
scoring/monitor.py         # live quality and drift checks
tests/test_state.py        # state function with a fake GroupState
tests/test_streaming_state.py  # Spark: restart equals batch (marker: spark)
tests/test_monitor.py
docs/adr/0007..0009
```
