"""
Export a small, version-controlled sample of scored predictions for the
deployed dashboard.

Streamlit Community Cloud runs neither Docker, Spark, nor a local Redpanda
broker -- it only executes plain Python against files already in the repo.
Mirroring Project 1's approach (data/gold_sample/windows.parquet), this
script reads the full local data/predictions/ (gitignored, can be large)
and writes a small, representative sample to data/predictions_sample/,
which IS committed to the repo. dashboard/app.py falls back to this sample
whenever the live predictions folder is empty or missing.

Usage (after running the full local pipeline for a while):
    python scoring/export_sample.py
"""

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "data", "predictions")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "predictions_sample")
SAMPLE_PATH = os.path.join(SAMPLE_DIR, "predictions_sample.parquet")

MAX_ROWS = 8000  # small enough to commit to git, large enough to fill the charts


def main():
    if not os.path.isdir(PREDICTIONS_DIR):
        print(f"No predictions found at {PREDICTIONS_DIR}. Run the scoring "
              f"pipeline first (see README).")
        return

    df = pd.read_parquet(PREDICTIONS_DIR)
    df = df.drop_duplicates(subset=["transaction_id"]).sort_values("event_time")
    print(f"Loaded {len(df):,} scored transactions.")

    if len(df) > MAX_ROWS:
        # Keep every flagged transaction (alerts are the interesting part of
        # the dashboard) plus a random sample of the rest, up to MAX_ROWS.
        alerts = df[df["predicted_fraud"]]
        rest = df[~df["predicted_fraud"]]
        remaining_budget = max(MAX_ROWS - len(alerts), 0)
        rest_sample = rest.sample(n=min(remaining_budget, len(rest)), random_state=42)
        df = pd.concat([alerts, rest_sample]).sort_values("event_time")
        print(f"Sampled down to {len(df):,} rows ({len(alerts):,} alerts kept in full).")

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    df.to_parquet(SAMPLE_PATH, index=False)
    print(f"Wrote sample to {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
