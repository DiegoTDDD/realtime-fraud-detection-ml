"""Spark: the scoring query keeps per-card state across a restart and writes each batch once.

Marked `spark` (needs Java and downloads the Delta and Kafka jars on first run). The pipeline
check runs it with `pytest -m spark`; the unit-test job skips it.
"""

import json
import os
import time

import pandas as pd
import pytest
from conftest import START, tx

pytest.importorskip("pyspark")
pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    import score_stream

    session = score_stream.build_spark("test-scoring", master="local[2]", kafka=False)
    yield session
    session.stop()


def events():
    """Two cards with bursts that straddle the restart, and a third card that starts late."""
    rows = []
    for i in range(30):
        rows.append(tx("A", i * 7))
        rows.append(tx("B", i * 23, city="Recife"))
    rows += [tx("C", 150 + i, category="entertainment", amount=2.0) for i in range(10)]
    return sorted(rows, key=lambda r: r["timestamp"])


def write_files(folder, rows, start, size=8):
    """Files the source reads one per micro-batch, oldest first (it orders by modification time)."""
    base = time.time() - 1000
    for n, i in enumerate(range(0, len(rows), size)):
        path = folder / f"part-{start + n:04d}.json"
        with open(path, "w") as f:
            for row in rows[i : i + size]:
                f.write(json.dumps(row) + "\n")
        os.utime(path, (base + start + n, base + start + n))
    return start + (len(rows) + size - 1) // size


def scorer():
    import numpy as np
    import score_stream
    from sklearn.dummy import DummyClassifier

    cols = ["amount", "distance_km_from_home", "tx_count_last_5min", "time_since_last_tx"]
    model = DummyClassifier(strategy="prior").fit(np.zeros((4, len(cols))), [0, 1, 0, 0])
    return score_stream.make_scorer(model, cols, 0.5, {"A": "Sao Paulo", "B": "Recife"})


def run_once(spark, source, out, checkpoint):
    import score_stream
    from pyspark.sql import functions as F

    stream = (
        spark.readStream.schema(score_stream.EVENT_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .json(str(source))
        .withColumn("kafka_partition", F.lit(0))
        .withColumn("kafka_offset", F.lit(0).cast("long"))
        .withColumn("event_time", F.to_timestamp("timestamp"))
    )
    score_stream.start(
        stream, scorer(), str(out), str(checkpoint), available_now=True
    ).awaitTermination()


def test_velocity_survives_a_restart_and_matches_the_batch_features(spark, tmp_path):
    from features import add_velocity_features

    source, out, checkpoint = tmp_path / "in", tmp_path / "out", tmp_path / "chk"
    source.mkdir()
    rows = events()
    half = len(rows) // 2
    next_file = write_files(source, rows[:half], 0)
    run_once(spark, source, out, checkpoint)  # first run, then the job "restarts"
    write_files(source, rows[half:], next_file)
    run_once(spark, source, out, checkpoint)

    got = spark.read.format("delta").load(str(out)).toPandas()
    assert len(got) == len(rows) and got["transaction_id"].is_unique
    expected = add_velocity_features(pd.DataFrame(rows))
    merged = got.merge(expected, on="transaction_id", suffixes=("", "_batch"))
    for name in ["tx_count_last_5min", "time_since_last_tx"]:
        assert (merged[name] == merged[f"{name}_batch"]).all(), name
    assert merged["tx_count_last_5min"].max() > 10  # the bursts were counted across files


def test_a_replayed_batch_is_not_written_twice(spark, tmp_path):
    import score_stream

    df = spark.createDataFrame(pd.DataFrame({"x": [1, 2, 3]}))
    path = str(tmp_path / "t")
    for _ in range(2):
        score_stream.write_predictions(df, path, "app", 7)
    score_stream.write_predictions(df, path, "app", 8)
    assert spark.read.format("delta").load(path).count() == 6


def test_the_parser_reads_the_wire_format_and_plain_json(spark):
    import score_stream

    payload = json.dumps(tx("A", 1)).encode()
    raw = spark.createDataFrame(
        [(b"\x00\x00\x00\x00\x01" + payload, 0, 10), (payload, 1, 11), (b"not json", 0, 12)],
        "value binary, partition int, offset long",
    )
    got = score_stream.parse_events(raw).toPandas()
    assert got["card_id"].tolist() == ["A", "A"]
    assert got["kafka_offset"].tolist() == [10, 11]
    assert got["event_time"].iloc[0] == pd.Timestamp(START).tz_localize(None) + pd.Timedelta(
        seconds=1
    )
