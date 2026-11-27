# 0008. Per-card state in Spark's state store; predictions in Delta, written once per batch

**Status:** Accepted (supersedes ADR 0003)

## Context
The velocity features need each card's history across micro-batches. ADR 0003 kept it in a dict inside the scoring process: lost on every restart (the first five minutes after a redeploy saw no card-testing bursts) and impossible to spread across workers. Predictions were appended to Parquet, so a micro-batch replayed after a crash was written twice.

Reviewing the old code also found a training/serving skew: serving reset `time_since_last_tx` to the "no recent transaction" value once the previous transaction was more than five minutes old, while training used the real gap.

## Decision
- `applyInPandasWithState`, grouped by `card_id`, runs `scoring/velocity_state.py:update_card`. The state per card is the transaction times inside the 5-minute window plus the time of the last transaction. Spark checkpoints it with the Kafka offsets, so state and input position always match after a restart, and each card's state lives on the worker that owns its key.
- A processing-time timeout drops a card idle for 24 hours. Training caps `time_since_last_tx` at the same 24 hours, so both paths still agree. Empty micro-batches are disabled (`spark.sql.streaming.noDataMicroBatches.enabled=false`); timeouts fire with the next data batch.
- The producer keys messages by `card_id`, so a card's events stay in order within one partition, which the window logic relies on.
- Each scored micro-batch is appended to the Delta table `data/predictions` with `txnAppId` = the query id from the checkpoint and `txnVersion` = the batch id. Delta skips a (app id, version) it has already committed, so a replayed batch is not written twice. The id comes from the checkpoint, so a new checkpoint (batch ids from 0 again) cannot be mistaken for an old one.
- Scoring runs in `mapInPandas` on the workers instead of collecting the batch to the driver.

## Alternatives considered
- **Redis.** A second service to run and keep consistent with the Kafka offsets; Spark's state store gives that consistency for free.
- **Keeping the dict and replaying five minutes of the topic on start.** Fixes restarts, not multiple workers.

## Consequences
- `tests/test_streaming_state.py` (Spark) runs the query, stops it, adds data, restarts it from the same checkpoint and checks that every velocity value equals the batch computation over all events, and that a repeated (app id, version) write adds nothing.
- The pipeline check kills the scoring job with SIGKILL mid-stream, restarts it, and requires exactly one prediction per Kafka message.
