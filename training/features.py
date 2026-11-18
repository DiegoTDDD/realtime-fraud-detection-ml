"""Feature engineering shared by training (batch) and real-time scoring (streaming).

Training and serving import the same functions from this module, so the model
sees features computed the same way in both places. Before this module existed
the two copies had drifted: training measured time in whole seconds while
serving used fractional seconds, so a transaction 300.4 s after the previous
one fell outside the 5-minute window at serving time but inside it at training
time. tests/test_features.py now checks that both paths produce identical
features for the same transactions.

Features, and the fraud pattern each one targets:
  - amount                 -> high_amount
  - distance_km_from_home  -> impossible_travel (see ADR 0002 for the 3 options tried)
  - tx_count_last_5min     -> card_testing bursts
  - time_since_last_tx     -> card_testing bursts
  - hour_of_day, category  -> general behaviour
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "distance_km_from_home",
    "tx_count_last_5min",
    "time_since_last_tx",
    "hour_of_day",
]
CATEGORY_COL = "category"  # one-hot encoded separately
VELOCITY_WINDOW_S = 300  # 5 minutes
NO_RECENT_TX = 99999.0  # time_since_last_tx for a card's first transaction
EARTH_RADIUS_KM = 6371

# Public coordinates of the 10 cities used by the producer in realtime-fraud-streaming.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Sao Paulo": (-23.5505, -46.6333),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Belo Horizonte": (-19.9167, -43.9345),
    "Curitiba": (-25.4284, -49.2733),
    "Porto Alegre": (-30.0346, -51.2177),
    "Salvador": (-12.9777, -38.5016),
    "Recife": (-8.0476, -34.8770),
    "Fortaleza": (-3.7172, -38.5433),
    "Brasilia": (-15.7939, -47.8828),
    "Manaus": (-3.1190, -60.0217),
}


def haversine_km(city_a: str, city_b: str) -> float:
    """Great-circle distance in km between two of the known cities."""
    if city_a == city_b:
        return 0.0
    lat1, lon1, lat2, lon2 = map(np.radians, [*CITY_COORDS[city_a], *CITY_COORDS[city_b]])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def to_epoch_seconds(timestamps: pd.Series) -> np.ndarray:
    """Whole seconds since the epoch. Both training and serving use this resolution."""
    return (
        pd.to_datetime(timestamps, format="ISO8601").values.astype("datetime64[s]").astype(np.int64)
    )


# --------------------------------------------------------------------------- batch (training)


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add tx_count_last_5min and time_since_last_tx per card.

    tx_count_last_5min counts the card's earlier transactions at most
    VELOCITY_WINDOW_S seconds before this one. A two-pointer window per card
    keeps this O(n) instead of the O(n^2) of comparing every pair.
    Returns a copy sorted by (card_id, timestamp).
    """
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    tx_count = np.zeros(len(df), dtype=int)
    time_since_last = np.full(len(df), NO_RECENT_TX)

    for _, group in df.groupby("card_id"):
        idx = group.index.values
        times = to_epoch_seconds(group["timestamp"])
        left = 0
        for i in range(len(times)):
            while times[i] - times[left] > VELOCITY_WINDOW_S:
                left += 1
            tx_count[idx[i]] = i - left
            if i > 0:
                time_since_last[idx[i]] = times[i] - times[i - 1]

    df["tx_count_last_5min"] = tx_count
    df["time_since_last_tx"] = time_since_last
    return df


def get_home_city_map(df: pd.DataFrame) -> dict[str, str]:
    """Each card's home city, inferred as the city it transacts in most often.

    The event stream has no home-city field, so it has to be inferred. Training
    calls this on the training split only, and the mapping is saved to
    training/card_home_city.json for the scoring job to use unchanged.
    Ties go to the alphabetically first city, so the result is deterministic.
    """
    return df.groupby("card_id")["city"].agg(lambda s: s.mode().iloc[0]).to_dict()


def add_distance_feature(df: pd.DataFrame, home_city: Mapping[str, str]) -> pd.DataFrame:
    """Add distance_km_from_home; 0 for a card with no known home city (as in serving)."""
    df = df.copy()
    homes = df["card_id"].map(home_city)
    df["distance_km_from_home"] = [
        haversine_km(city, home) if isinstance(home, str) else 0.0
        for city, home in zip(df["city"], homes, strict=True)
    ]
    return df


def add_category_columns(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode the category. With feature_cols, add any missing column as 0.

    A micro-batch rarely contains every category, so serving passes the
    model's feature list to make sure every column the model expects exists.
    """
    dummies = pd.get_dummies(df[CATEGORY_COL], prefix=CATEGORY_COL)
    df = pd.concat([df, dummies], axis=1)
    for name in feature_cols or []:
        if name not in df.columns:
            df[name] = 0
    return df


def build_features(
    df: pd.DataFrame, home_city: Mapping[str, str]
) -> tuple[pd.DataFrame, list[str]]:
    """Batch features for training. Returns the frame and the model's feature columns."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["hour_of_day"] = df["timestamp"].dt.hour
    df = add_velocity_features(df)
    df = add_distance_feature(df, home_city)
    df = add_category_columns(df)
    categories = sorted(c for c in df.columns if c.startswith(f"{CATEGORY_COL}_"))
    return df, FEATURE_COLUMNS + categories


# --------------------------------------------------------------------------- streaming (serving)


class CardHistory:
    """Recent transaction times per card, kept across micro-batches.

    Process-local: correct only while the scoring job runs as a single process
    (see ADR 0003). A multi-worker deployment would keep this in a shared state
    store such as Redis or Spark's stateful operators.
    """

    def __init__(self) -> None:
        self._times: dict[str, deque[int]] = {}

    def update(self, card_id: str, epoch_s: int) -> tuple[int, float]:
        """Record a transaction; return (tx_count_last_5min, time_since_last_tx)."""
        history = self._times.setdefault(card_id, deque())
        while history and epoch_s - history[0] > VELOCITY_WINDOW_S:
            history.popleft()
        count = len(history)
        since_last = float(epoch_s - history[-1]) if history else NO_RECENT_TX
        history.append(epoch_s)
        return count, since_last


def featurize_batch(
    pdf: pd.DataFrame,
    home_city: Mapping[str, str],
    history: CardHistory,
    feature_cols: list[str],
    time_col: str = "event_time",
) -> pd.DataFrame:
    """Streaming features for one micro-batch, in arrival order.

    Returns the batch sorted by time with every column in feature_cols present.
    """
    pdf = pdf.sort_values(time_col, kind="stable").reset_index(drop=True)
    seconds = to_epoch_seconds(pdf[time_col])
    velocity = [
        history.update(card, int(t)) for card, t in zip(pdf["card_id"], seconds, strict=True)
    ]
    pdf["tx_count_last_5min"] = [count for count, _ in velocity]
    pdf["time_since_last_tx"] = [since for _, since in velocity]
    pdf["hour_of_day"] = pd.to_datetime(pdf[time_col], format="ISO8601").dt.hour
    pdf = add_distance_feature(pdf, home_city)
    return add_category_columns(pdf, feature_cols)
