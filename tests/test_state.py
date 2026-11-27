"""The per-card state function the streaming job gives applyInPandasWithState."""

import pandas as pd
import pytest
from conftest import tx
from features import NO_RECENT_TX, STATE_TTL_S, add_velocity_features
from velocity_state import VELOCITY_COLUMNS, update_card


class FakeState:
    """The parts of Spark's GroupState that update_card uses."""

    def __init__(self, timed_out: bool = False) -> None:
        self.value = None
        self.hasTimedOut = timed_out
        self.timeout_ms = None

    @property
    def exists(self) -> bool:
        return self.value is not None

    @property
    def get(self):
        return self.value

    def update(self, value) -> None:
        self.value = value

    def remove(self) -> None:
        self.value = None

    def setTimeoutDuration(self, ms: int) -> None:
        self.timeout_ms = ms


def frame(rows):
    pdf = pd.DataFrame(rows)
    pdf["event_time"] = pd.to_datetime(pdf["timestamp"]).dt.tz_localize(None)
    pdf["kafka_partition"] = 0
    pdf["kafka_offset"] = range(len(pdf))
    return pdf


def run(state, rows):
    return pd.concat(update_card(("A",), iter([frame(rows)]), state), ignore_index=True)


def test_state_carries_history_across_micro_batches():
    seconds = [0, 10, 20, 400, 410, 900, 900 + STATE_TTL_S + 1]
    rows = [tx("A", s) for s in seconds]
    state = FakeState()
    parts = [run(state, rows[:2]), run(state, rows[2:4]), run(state, rows[4:])]
    streamed = pd.concat(parts, ignore_index=True)
    batch = add_velocity_features(pd.DataFrame(rows))
    assert streamed[VELOCITY_COLUMNS].values.tolist() == batch[VELOCITY_COLUMNS].values.tolist()
    assert streamed["time_since_last_tx"].tolist()[3] == 380  # older than the 5-minute window
    assert streamed["time_since_last_tx"].tolist()[-1] == NO_RECENT_TX  # idle past the TTL
    assert state.timeout_ms == STATE_TTL_S * 1000


def test_rows_arriving_out_of_order_inside_a_batch_are_sorted_by_time():
    out = run(FakeState(), [tx("A", 30), tx("A", 0), tx("A", 10)])
    assert out["tx_count_last_5min"].tolist() == [0, 1, 2]


def test_a_timed_out_card_starts_again_with_no_history():
    state = FakeState()
    run(state, [tx("A", 0)])
    state.hasTimedOut = True
    assert list(update_card(("A",), iter([]), state)) == []
    assert not state.exists
    state.hasTimedOut = False
    out = run(state, [tx("A", 5)])
    assert out["time_since_last_tx"].tolist() == [NO_RECENT_TX]


@pytest.mark.parametrize("gap", [299, 300, 301, STATE_TTL_S, STATE_TTL_S + 1])
def test_window_and_ttl_boundaries_match_the_batch_rule(gap):
    rows = [tx("A", 0), tx("A", gap)]
    state = FakeState()
    streamed = pd.concat([run(state, rows[:1]), run(state, rows[1:])], ignore_index=True)
    batch = add_velocity_features(pd.DataFrame(rows))
    assert streamed[VELOCITY_COLUMNS].values.tolist() == batch[VELOCITY_COLUMNS].values.tolist()
