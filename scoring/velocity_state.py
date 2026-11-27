"""Per-card velocity state for the streaming job, kept in Spark's state store.

`update_card` is the function given to applyInPandasWithState, grouped by card_id. Spark keeps
each card's state (recent transaction times and the time of its last transaction) next to the
Kafka offsets in the checkpoint, so a restart resumes with the same history and input position,
and each card's state lives with its key on whichever worker owns it (ADR 0008).

No pyspark import at module level: the unit tests call update_card with a fake state object.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
from features import STATE_TTL_S, to_epoch_seconds, velocity_for_card

INPUT_COLUMNS = [
    "transaction_id",
    "card_id",
    "amount",
    "merchant",
    "category",
    "city",
    "timestamp",
    "event_time",
    "is_fraud",
    "fraud_type",
    "kafka_partition",
    "kafka_offset",
]
VELOCITY_COLUMNS = ["tx_count_last_5min", "time_since_last_tx"]
OUTPUT_DDL = (
    "transaction_id string, card_id string, amount double, merchant string, category string, "
    "city string, timestamp string, event_time timestamp, is_fraud boolean, fraud_type string, "
    "kafka_partition int, kafka_offset long, tx_count_last_5min int, time_since_last_tx double"
)
STATE_DDL = "window array<long>, last long"
STATE_TTL_MS = STATE_TTL_S * 1000


def update_card(key: tuple, batches: Iterator[pd.DataFrame], state: Any) -> Iterator[pd.DataFrame]:
    """Velocity features for one card's transactions in this micro-batch, updating its state."""
    if state.hasTimedOut:
        state.remove()  # idle for STATE_TTL_S: the card counts as new from now on
        return
    pdf = pd.concat(list(batches), ignore_index=True)
    if pdf.empty:
        return
    pdf = pdf.sort_values("timestamp", kind="stable").reset_index(drop=True)
    window, last = (list(state.get[0]), state.get[1]) if state.exists else ([], None)
    counts, since, window, last = velocity_for_card(
        to_epoch_seconds(pdf["timestamp"]), window, last
    )
    state.update((window, last))
    state.setTimeoutDuration(STATE_TTL_MS)
    pdf["tx_count_last_5min"] = counts
    pdf["time_since_last_tx"] = since
    yield pdf[INPUT_COLUMNS + VELOCITY_COLUMNS]
