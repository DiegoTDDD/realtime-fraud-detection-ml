"""Real-time fraud scoring: Kafka -> per-card state -> XGBoost -> Delta.

Reads the `transactions` topic of realtime-fraud-streaming as an independent consumer, scores
each transaction with the model from training/train_model.py and writes the predictions to a
Delta table, data/predictions.

  1. Parse: the producer writes the schema registry's wire format (a 0 byte and a 4-byte schema
     id before the JSON). The header is stripped; plain JSON is still accepted.
  2. Velocity features (tx_count_last_5min, time_since_last_tx) need each card's history across
     micro-batches. applyInPandasWithState keeps it in Spark's state store, checkpointed with
     the Kafka offsets, so a restart loses no history (ADR 0008).
  3. Score each micro-batch with the shared feature code (training/features.py) and append it
     to Delta with the batch id as transaction version, so a batch replayed after a crash is
     not written twice.

Run (Redpanda up, producer running, model trained):
    python scoring/score_stream.py
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Spark starts Python workers for the pandas functions; point them at this interpreter. On
# Windows plain "python" can resolve to the Microsoft Store stub instead.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if platform.system() == "Windows":
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ["PATH"] = (
        os.path.join(os.environ["HADOOP_HOME"], "bin") + os.pathsep + os.environ["PATH"]
    )

from delta import configure_spark_with_delta_pip  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.streaming import StreamingQuery  # noqa: E402
from pyspark.sql.streaming.state import GroupStateTimeout  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

ROOT = Path(__file__).resolve().parent.parent
for folder in ("training", "scoring"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))
from features import add_stream_features  # noqa: E402
from velocity_state import OUTPUT_DDL, STATE_DDL, update_card  # noqa: E402

log = logging.getLogger(__name__)
PREDICTIONS_DIR = ROOT / "data" / "predictions"
CHECKPOINT_DIR = ROOT / "data" / "_checkpoints" / "predictions"
MODEL_PATH = ROOT / "training" / "model.joblib"
HOME_CITY_PATH = ROOT / "training" / "card_home_city.json"

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")
TOPIC = "transactions"
KAFKA_PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8"
TRIGGER = "10 seconds"
JAVA_OPTIONS = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED -Dio.netty.tryReflectionSetAccessible=true"
)

EVENT_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("card_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("merchant", StringType()),
        StructField("category", StringType()),
        StructField("city", StringType()),
        StructField("timestamp", StringType()),
        StructField("is_fraud", BooleanType()),
        StructField("fraud_type", StringType()),
    ]
)
PREDICTION_DDL = OUTPUT_DDL + ", fraud_probability double, predicted_fraud boolean"
SHIPPED_MODULES = [ROOT / "training" / "features.py", ROOT / "scoring" / "velocity_state.py"]


def build_spark(
    app: str = "realtime-fraud-scoring", master: str | None = None, kafka: bool = True
) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        # Processing-time timeouts would otherwise make Spark run an empty micro-batch after
        # every batch; idle cards are expired when the next data batch arrives instead.
        .config("spark.sql.streaming.noDataMicroBatches.enabled", "false")
        .config("spark.databricks.delta.snapshotPartitions", "2")
        # Arrow (used by the pandas functions) needs this on Java 17 and later.
        .config("spark.driver.extraJavaOptions", JAVA_OPTIONS)
        .config("spark.executor.extraJavaOptions", JAVA_OPTIONS)
    )
    if master:
        builder = builder.master(master)
    spark = configure_spark_with_delta_pip(
        builder, extra_packages=[KAFKA_PKG] if kafka else []
    ).getOrCreate()
    for module in SHIPPED_MODULES:  # the pandas functions run in worker processes
        spark.sparkContext.addPyFile(str(module))
    return spark


def parse_events(raw: DataFrame) -> DataFrame:
    """Kafka records -> transaction columns, lineage (partition, offset) and event_time."""
    has_header = (F.length("value") > 5) & (F.hex(F.substring("value", 1, 1)) == "00")
    payload = F.when(has_header, F.expr("substring(value, 6, length(value) - 5)")).otherwise(
        F.col("value")
    )
    return (
        raw.select(
            F.from_json(payload.cast("string"), EVENT_SCHEMA).alias("t"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
        )
        .select("t.*", "kafka_partition", "kafka_offset")
        .where(F.col("transaction_id").isNotNull() & F.col("card_id").isNotNull())
        .withColumn("event_time", F.to_timestamp("timestamp"))
    )


def with_velocity(events: DataFrame) -> DataFrame:
    """Add the velocity features from each card's state (see scoring/velocity_state.py)."""
    return events.groupBy("card_id").applyInPandasWithState(
        update_card,
        outputStructType=OUTPUT_DDL,
        stateStructType=STATE_DDL,
        outputMode="append",
        timeoutConf=GroupStateTimeout.ProcessingTimeTimeout,
    )


def make_scorer(
    model: Any, feature_cols: list[str], threshold: float, home_city: dict[str, str]
) -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:
    """A mapInPandas function adding fraud_probability and predicted_fraud."""
    columns = [c.split(" ")[0] for c in PREDICTION_DDL.split(", ")]

    def score(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in batches:
            if pdf.empty:
                continue
            features = add_stream_features(pdf, home_city, feature_cols)
            pdf["fraud_probability"] = model.predict_proba(features[feature_cols])[:, 1]
            pdf["predicted_fraud"] = pdf["fraud_probability"] >= threshold
            yield pdf[columns]

    return score


def write_predictions(df: DataFrame, path: str, app_id: str, batch_id: int) -> None:
    """Append to Delta once per (app_id, batch_id): a replayed batch is skipped by Delta."""
    (
        df.write.format("delta")
        .mode("append")
        .option("txnAppId", app_id)
        .option("txnVersion", batch_id)
        .save(path)
    )


def query_app_id(checkpoint_dir: str) -> str:
    """The stream's id from its checkpoint: a new checkpoint (batch ids from 0 again) gets a new
    id, so Delta does not mistake its first batches for ones it has already written."""
    with open(os.path.join(checkpoint_dir, "metadata"), encoding="utf-8") as f:
        return "fraud_scoring-" + json.loads(f.readline())["id"]


def start(
    events: DataFrame,
    scorer: Callable,
    output: str,
    checkpoint: str,
    available_now: bool = False,
) -> StreamingQuery:
    """Velocity state, scoring and the idempotent Delta sink, as one streaming query."""
    app_id: list[str] = []

    def sink(batch: DataFrame, batch_id: int) -> None:
        if not app_id:
            app_id.append(query_app_id(checkpoint))
        scored = batch.mapInPandas(scorer, PREDICTION_DDL)
        write_predictions(scored, output, app_id[0], batch_id)
        log.info("[batch %d] written", batch_id)

    writer = (
        with_velocity(events)
        .writeStream.foreachBatch(sink)
        .option("checkpointLocation", checkpoint)
    )
    writer = (
        writer.trigger(availableNow=True)
        if available_now
        else writer.trigger(processingTime=TRIGGER)
    )
    return writer.queryName("fraud_scoring").start()


def load_artifacts() -> tuple:
    artifact = joblib.load(MODEL_PATH)
    home_city = json.loads(HOME_CITY_PATH.read_text(encoding="utf-8"))
    log.info(
        "Model: %d features, threshold %.4f", len(artifact["feature_cols"]), artifact["threshold"]
    )
    return artifact["model"], artifact["feature_cols"], artifact["threshold"], home_city


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model, feature_cols, threshold, home_city = load_artifacts()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )
    query = start(
        parse_events(raw),
        make_scorer(model, feature_cols, threshold, home_city),
        str(PREDICTIONS_DIR),
        str(CHECKPOINT_DIR),
    )
    log.info("Scoring stream started. Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
