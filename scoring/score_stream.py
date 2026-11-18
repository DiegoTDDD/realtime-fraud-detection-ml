"""
Real-time fraud scoring job for Project 6.

Reads the same Redpanda `transactions` topic Project 1's streaming job
reads (this is the "additional consumer" decided in the architecture
discussion), scores each transaction with the offline-trained XGBoost
model (training/model.joblib), and writes an augmented Gold dataset
(fraud_probability, predicted_fraud) to its own path -- kept separate from
Project 1's Gold so the two projects don't collide.

Feature engineering here must exactly mirror training/train_model.py, or
the model sees a different feature distribution at serving time than it
was trained on (training-serving skew). Two features need special care
because they depend on data *outside* a single micro-batch:

  - distance_km_from_home: uses training/card_home_city.json, the same
    home-city mapping computed at training time (this doesn't change
    live -- "home city per card" is a stable identity attribute, not a
    real-time signal).
  - tx_count_last_5min / time_since_last_tx: need each card's recent
    transaction history, which can span multiple micro-batches. This job
    keeps that history in a process-local Python dict, updated inside
    foreachBatch. That's a documented simplification: it only works
    because this job runs as a single local process (matching how every
    project in this portfolio runs), not a distributed multi-worker
    cluster. A production system would back this with a proper state
    store (e.g. Spark's own stateful streaming operators, or Redis).

Feature code is imported from training/features.py, the same module training
uses, so the two cannot drift apart (tests/test_features.py checks it).

Run (inside the `fraud-ml` conda env, with Redpanda up -- Project 1's
streaming_job.py does NOT need to be running; this is an independent
consumer of the same topic):
    python scoring/score_stream.py
Stop with Ctrl+C.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from typing import Any

import joblib

# ---------------------------------------------------------------------------
# Tell Spark exactly which Python executable to use for worker processes.
# foreachBatch (via toPandas()) is the first thing in this pipeline that
# actually spawns a separate Python worker process -- Project 1's
# streaming_job.py never needed one, since it only uses native Spark SQL
# functions. Without this, Spark falls back to calling "python" via the
# system PATH, which on Windows can resolve to the Microsoft Store's
# App Execution Alias stub instead of the real conda interpreter, causing
# "Python worker failed to connect back" / socket timeout errors.
# ---------------------------------------------------------------------------
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ---------------------------------------------------------------------------
# Windows Hadoop shim -- identical to Project 1's streaming_job.py, since
# Spark needs the same winutils.exe + hadoop.dll to write files on Windows.
# ---------------------------------------------------------------------------
if platform.system() == "Windows":
    hadoop_home = r"C:\hadoop"
    os.environ.setdefault("HADOOP_HOME", hadoop_home)
    os.environ["PATH"] = os.path.join(hadoop_home, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
from features import CardHistory, featurize_batch  # noqa: E402

log = logging.getLogger(__name__)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "_checkpoints", "predictions")

MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "model.joblib")
HOME_CITY_PATH = os.path.join(PROJECT_ROOT, "training", "card_home_city.json")

KAFKA_BOOTSTRAP = "localhost:19092"
TOPIC = "transactions"
KAFKA_PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8"

# Schema of the JSON payload -- identical to Project 1's producer output.
SCHEMA = StructType(
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

# Explicit output schema. Without this, Spark infers types from the pandas
# data on each call to createDataFrame() -- which fails with
# CANNOT_DETERMINE_TYPE whenever a column is entirely null in a given
# micro-batch (e.g. fraud_type, when that batch happens to contain zero
# fraudulent transactions). Declaring the schema up front avoids relying
# on per-batch inference altogether.
OUTPUT_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("card_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("merchant", StringType()),
        StructField("category", StringType()),
        StructField("city", StringType()),
        StructField("timestamp", StringType()),
        StructField("event_time", TimestampType()),
        StructField("is_fraud", BooleanType()),
        StructField("fraud_type", StringType()),
        StructField("fraud_probability", DoubleType()),
        StructField("predicted_fraud", BooleanType()),
    ]
)

OUTPUT_COLS = [field.name for field in OUTPUT_SCHEMA.fields]

# Filled by load_artifacts() in main(), once before the stream starts, not per micro-batch.
MODEL: Any = None
FEATURE_COLS: list[str] = []
THRESHOLD = 0.5
HOME_CITY: dict[str, str] = {}
# Recent transaction times per card, across micro-batches. Process-local: see the
# module docstring for why this only works while the job runs as a single process.
CARD_HISTORY = CardHistory()


def load_artifacts() -> None:
    """Load the model, its feature list, its threshold and the home-city map."""
    global MODEL, FEATURE_COLS, THRESHOLD, HOME_CITY
    log.info("Loading model from %s...", MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    MODEL, FEATURE_COLS, THRESHOLD = (
        artifact["model"],
        artifact["feature_cols"],
        artifact["threshold"],
    )
    with open(HOME_CITY_PATH, encoding="utf-8") as f:
        HOME_CITY = json.load(f)
    log.info("Model loaded. %d features, threshold=%.4f", len(FEATURE_COLS), THRESHOLD)


def score_batch(batch_df: DataFrame, batch_id: int) -> None:
    """foreachBatch callback: score one Spark micro-batch with the offline
    XGBoost model and append the results to the predictions Parquet path.
    """
    pdf = batch_df.toPandas()
    if pdf.empty:
        return

    pdf = featurize_batch(pdf, HOME_CITY, CARD_HISTORY, FEATURE_COLS)

    X = pdf[FEATURE_COLS]
    pdf["fraud_probability"] = MODEL.predict_proba(X)[:, 1]
    pdf["predicted_fraud"] = (pdf["fraud_probability"] >= THRESHOLD).astype(bool)

    result = pdf[OUTPUT_COLS]

    (
        batch_df.sparkSession.createDataFrame(result, schema=OUTPUT_SCHEMA)
        .write.mode("append")
        .parquet(PREDICTIONS_DIR)
    )

    n_alerts = int(result["predicted_fraud"].sum())
    log.info("[batch %d] scored %d tx, %d alerts", batch_id, len(result), n_alerts)


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("realtime-fraud-scoring")
        .config("spark.jars.packages", KAFKA_PKG)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_artifacts()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    parsed = (
        raw.select(from_json(col("value").cast("string"), SCHEMA).alias("t"))
        .select("t.*")
        .withColumn("event_time", to_timestamp(col("timestamp")))
    )

    query = (
        parsed.writeStream.foreachBatch(score_batch)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .queryName("fraud_scoring")
        .start()
    )

    log.info("Scoring stream started. Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
