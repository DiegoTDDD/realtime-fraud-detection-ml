# 0003. Keep per-card velocity state in the scoring process

**Status:** Accepted, with a stated limit

## Context
`tx_count_last_5min` and `time_since_last_tx` need each card's recent history, which spans several micro-batches. The scoring job uses `foreachBatch` with a pandas model call, so Spark's own stateful operators would not hold this state.

## Decision
A `CardHistory` object in the scoring process keeps a deque of recent transaction times per card. It drops entries older than five minutes on every update (`training/features.py`).

## Alternatives considered
- **Spark stateful operators (`applyInPandasWithState`).** Distributed and checkpointed, but much more code, and they would split the model call across workers.
- **Redis.** Survives restarts and works with many workers, but adds a service to run, and the portfolio deliberately runs locally at no cost.

## Consequences
- Simple and fast. Memory grows only with the number of cards active in the last five minutes.
- Correct only with a single scoring process. A restart resets the history, so the first five minutes after it undercount velocity. Moving to Redis or Spark state is the step to take before running more than one worker.
