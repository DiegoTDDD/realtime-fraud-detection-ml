"""
Export a small, version-controlled sample of scored predictions for the
deployed dashboard.

Streamlit Community Cloud runs neither Docker, Spark, nor a local Redpanda
broker -- it only executes plain Python against files already in the repo.
Mirroring Project 1's approach (data/gold_sample/windows.parquet), this
script reads the full local data/predictions Delta table (gitignored, can be large)
and writes a small, representative sample to data/predictions_sample/,
which IS committed to the repo. dashboard/app.py falls back to this sample
whenever the live predictions folder is empty or missing.

Usage (after running the full local pipeline for a while):
    python scoring/export_sample.py
"""

import logging
import os

from deltalake import DeltaTable

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "data", "predictions")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "predictions_sample")
SAMPLE_PATH = os.path.join(SAMPLE_DIR, "predictions_sample.parquet")

MAX_ROWS = 8000  # small enough to commit to git, large enough to fill the charts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not os.path.isdir(os.path.join(PREDICTIONS_DIR, "_delta_log")):
        log.warning(
            "No predictions found at %s. Run the scoring pipeline first (see README).",
            PREDICTIONS_DIR,
        )
        return

    df = DeltaTable(PREDICTIONS_DIR).to_pandas()
    df = df.drop_duplicates(subset=["transaction_id"]).sort_values("event_time")
    log.info("Loaded %s scored transactions.", f"{len(df):,}")

    if len(df) > MAX_ROWS:
        # The latest MAX_ROWS transactions, contiguous in time. Keeping every alert and
        # sampling the rest would drop missed frauds more often than caught ones and
        # overstate the dashboard's live recall.
        df = df.tail(MAX_ROWS)
        log.info("Kept the latest %s transactions.", f"{len(df):,}")

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    df.to_parquet(SAMPLE_PATH, index=False)
    log.info("Wrote sample to %s", SAMPLE_PATH)


if __name__ == "__main__":
    main()
