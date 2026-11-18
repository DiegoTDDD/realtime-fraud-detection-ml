"""Feature engineering, and parity between the training and serving code paths."""

import pandas as pd
import pytest
from conftest import tx
from features import (
    NO_RECENT_TX,
    CardHistory,
    add_category_columns,
    add_distance_feature,
    add_velocity_features,
    build_features,
    featurize_batch,
    get_home_city_map,
    haversine_km,
)


def test_haversine_known_distance_and_symmetry():
    assert haversine_km("Sao Paulo", "Sao Paulo") == 0.0
    sp_rio = haversine_km("Sao Paulo", "Rio de Janeiro")
    assert sp_rio == pytest.approx(361, abs=5)  # published great-circle distance
    assert sp_rio == haversine_km("Rio de Janeiro", "Sao Paulo")
    assert haversine_km("Porto Alegre", "Manaus") > 3000


def test_velocity_counts_earlier_transactions_within_five_minutes():
    df = pd.DataFrame([tx("A", s) for s in (0, 100, 300, 301)] + [tx("B", 50)])
    out = add_velocity_features(build_ts(df)).set_index("transaction_id")
    assert list(out.loc[["A-0", "A-100", "A-300", "A-301"], "tx_count_last_5min"]) == [0, 1, 2, 2]
    assert list(out.loc[["A-0", "A-100", "A-300", "A-301"], "time_since_last_tx"]) == [
        NO_RECENT_TX,
        100,
        200,
        1,
    ]
    assert out.loc["B-50", "tx_count_last_5min"] == 0  # other cards do not count


def test_card_history_matches_batch_rule():
    history = CardHistory()
    assert [history.update("A", s) for s in (0, 100, 300, 301)] == [
        (0, NO_RECENT_TX),
        (1, 100.0),
        (2, 200.0),
        (2, 1.0),
    ]


def test_fractional_seconds_are_truncated_the_same_way_in_both_paths():
    # 300.4 s apart in real time, 300 s apart in whole seconds: inside the window in both paths.
    df = build_ts(pd.DataFrame([tx("A", 0.9), tx("A", 300.4)]))
    batch = add_velocity_features(df)
    stream = featurize_batch(df.rename(columns={"timestamp": "event_time"}), {}, CardHistory(), [])
    assert list(batch["tx_count_last_5min"]) == list(stream["tx_count_last_5min"]) == [0, 1]


def test_home_city_is_the_most_frequent_city_with_deterministic_ties():
    df = pd.DataFrame(
        [
            tx("A", 1, "Recife"),
            tx("A", 2, "Recife"),
            tx("A", 3, "Manaus"),
            tx("B", 1, "Salvador"),
            tx("B", 2, "Curitiba"),
        ]
    )
    assert get_home_city_map(df) == {"A": "Recife", "B": "Curitiba"}


def test_distance_is_zero_for_home_city_and_unknown_cards():
    df = pd.DataFrame([tx("A", 1, "Recife"), tx("A", 2, "Manaus"), tx("Z", 3, "Manaus")])
    out = add_distance_feature(df, {"A": "Recife"})
    assert out["distance_km_from_home"].iloc[0] == 0.0
    assert out["distance_km_from_home"].iloc[1] > 2000
    assert out["distance_km_from_home"].iloc[2] == 0.0


def test_missing_categories_are_added_as_zero_for_serving():
    df = pd.DataFrame([tx("A", 1, category="fuel")])
    out = add_category_columns(df, ["category_fuel", "category_travel"])
    assert out.loc[0, "category_fuel"] == 1
    assert out.loc[0, "category_travel"] == 0


def test_build_features_returns_numeric_features_in_a_stable_order(generated):
    df, cols = build_features(generated, get_home_city_map(generated))
    assert cols[:5] == [
        "amount",
        "distance_km_from_home",
        "tx_count_last_5min",
        "time_since_last_tx",
        "hour_of_day",
    ]
    assert cols[5:] == sorted(cols[5:]) and all(c.startswith("category_") for c in cols[5:])
    assert df[cols].notna().all().all()


@pytest.mark.parametrize("batch_size", [1, 37, 500])
def test_serving_features_equal_training_features(generated, batch_size):
    """The scoring job, fed the stream in micro-batches, must reproduce the training features."""
    home = get_home_city_map(generated)
    train_df, cols = build_features(generated, home)

    stream = generated.copy()
    stream["event_time"] = pd.to_datetime(stream["timestamp"], format="ISO8601")
    history = CardHistory()
    parts = [
        featurize_batch(stream.iloc[i : i + batch_size], home, history, cols)
        for i in range(0, len(stream), batch_size)
    ]
    served = pd.concat(parts).set_index("transaction_id")[cols]
    expected = train_df.set_index("transaction_id")[cols].loc[served.index]
    pd.testing.assert_frame_equal(served.astype(float), expected.astype(float))


def build_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df
