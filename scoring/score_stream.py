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

Run (inside the `fraud-ml` conda env, with Redpanda up -- Project 1's
streaming_job.py does NOT need to be running; this is an independent
consumer of the same topic):
    python scoring/score_stream.py
Stop with Ctrl+C.
"""

import json
import os
import platform
import sys
from collections import deque

import joblib
import numpy as np
import pandas as pd

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

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, BooleanType, TimestampType,
)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "_checkpoints", "predictions")

MODEL_PATH = os.path.join(PROJECT_ROOT, "training", "model.joblib")
HOME_CITY_PATH = os.path.join(PROJECT_ROOT, "training", "card_home_city.json")

KAFKA_BOOTSTRAP = "localhost:19092"
TOPIC = "transactions"
KAFKA_PKG = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"

# Schema of the JSON payload -- identical to Project 1's producer output.
SCHEMA = StructType([
    StructField("transaction_id", StringType()),
    StructField("card_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("merchant", StringType()),
    StructField("category", StringType()),
    StructField("city", StringType()),
    StructField("timestamp", StringType()),
    StructField("is_fraud", BooleanType()),
    StructField("fraud_type", StringType()),
])

# City coordinates -- must match training/train_model.py exactly, or
# distance_km_from_home would be computed differently at serving time
# than at training time.
CITY_COORDS = {
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
    if city_a == city_b:
        return 0.0
    lat1, lon1 = CITY_COORDS[city_a]
    lat2, lon2 = CITY_COORDS[city_b]
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# Explicit output schema. Without this, Spark infers types from the pandas
# data on each call to createDataFrame() -- which fails with
# CANNOT_DETERMINE_TYPE whenever a column is entirely null in a given
# micro-batch (e.g. fraud_type, when that batch happens to contain zero
# fraudulent transactions). Declaring the schema up front avoids relying
# on per-batch inference altogether.
OUTPUT_SCHEMA = StructType([
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
])

# ---------------------------------------------------------------------------
# Load model artifacts once, before the stream starts -- not per micro-batch.
# ---------------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH}...")
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
FEATURE_COLS = artifact["feature_cols"]
THRESHOLD = artifact["threshold"]
print(f"Model loaded. {len(FEATURE_COLS)} features, threshold={THRESHOLD:.4f}")

with open(HOME_CITY_PATH) as f:
    HOME_CITY = json.load(f)

# Process-local state: card_id -> deque of recent transaction epoch-seconds.
# Computes tx_count_last_5min / time_since_last_tx across micro-batches.
# See module docstring for why this only works in single-process local mode.
CARD_HISTORY: dict = {}


def score_batch(batch_df, batch_id: int):
    """foreachBatch callback: score one Spark micro-batch with the offline
    XGBoost model and append the results to the predictions Parquet path.
    """
    pdf = batch_df.toPandas()
    if pdf.empty:
        return

    # Process in arrival order so velocity features are computed
    # consistently within the batch.
    pdf = pdf.sort_values("event_time").reset_index(drop=True)

    distances = []
    tx_counts = []
    times_since_last = []

    for _, row in pdf.iterrows():
        card_id = row["card_id"]
        city = row["city"]
        ts = row["event_time"].timestamp()

        # --- distance_km_from_home ---
        home = HOME_CITY.get(card_id)
        distances.append(haversine_km(city, home) if home else 0.0)

        # --- velocity features, using process-local history ---
        history = CARD_HISTORY.setdefault(card_id, deque())
        while history and ts - history[0] > 300:  # 5-minute window
            history.popleft()
        tx_counts.append(len(history))
        times_since_last.append(ts - history[-1] if history else 99999.0)
        history.append(ts)

    pdf["distance_km_from_home"] = distances
    pdf["tx_count_last_5min"] = tx_counts
    pdf["time_since_last_tx"] = times_since_last
    pdf["hour_of_day"] = pdf["event_time"].dt.hour

    category_dummies = pd.get_dummies(pdf["category"], prefix="category")
    pdf = pd.concat([pdf, category_dummies], axis=1)

    # Ensure every feature column the model expects exists -- a category
    # that didn't appear in this micro-batch would otherwise be missing.
    for col_name in FEATURE_COLS:
        if col_name not in pdf.columns:
            pdf[col_name] = 0

    X = pdf[FEATURE_COLS]
    pdf["fraud_probability"] = MODEL.predict_proba(X)[:, 1]
    pdf["predicted_fraud"] = (pdf["fraud_probability"] >= THRESHOLD).astype(bool)

    output_cols = [
        "transaction_id", "card_id", "amount", "merchant", "category", "city",
        "timestamp", "event_time", "is_fraud", "fraud_type",
        "fraud_probability", "predicted_fraud",
    ]
    result = pdf[output_cols]

    (
        batch_df.sparkSession.createDataFrame(result, schema=OUTPUT_SCHEMA)
        .write.mode("append")
        .parquet(PREDICTIONS_DIR)
    )

    n_alerts = int(result["predicted_fraud"].sum())
    print(f"[batch {batch_id}] scored {len(result)} tx, {n_alerts} alerts")


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("realtime-fraud-scoring")
        .config("spark.jars.packages", KAFKA_PKG)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream
        .format("kafka")
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
        parsed.writeStream
        .foreachBatch(score_batch)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .queryName("fraud_scoring")
        .start()
    )

    print("Scoring stream started. Ctrl+C to stop.")
    query.awaitTermination()


if __name__ == "__main__":
    main()
