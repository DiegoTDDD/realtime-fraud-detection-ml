"""
Generate a training dataset for Project 6 (real-time fraud detection ML).

This script reuses the exact transaction-simulation logic from Project 1's
producer (producer/producer.py) -- same card population, same three fraud
patterns (high_amount, card_testing, impossible_travel), same ~2% base
fraud rate -- but runs it in batch, in memory, writing straight to Parquet.

Why batch instead of running the real producer through Kafka/Spark for
hours: training data generation and real-time streaming serve different
purposes. The streaming pipeline (Project 1) simulates a live payment
feed; this script simulates a historical dataset to train an offline
model on. Reusing the same generation logic keeps the two consistent
without paying the cost of running Kafka + Spark just to accumulate rows.

Usage:
    python training/generate_training_data.py
"""

import random
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_TRANSACTIONS = 150_000   # target dataset size
FRAUD_RATE = 0.02            # same base rate as Project 1's producer
OUTPUT_PATH = "data/training_raw.parquet"

random.seed(42)

CITIES = [
    "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba",
    "Porto Alegre", "Salvador", "Recife", "Fortaleza", "Brasilia", "Manaus",
]

CATEGORIES = {
    "groceries": (10, 120),
    "restaurant": (15, 200),
    "electronics": (100, 2000),
    "fashion": (30, 600),
    "travel": (200, 3000),
    "fuel": (40, 300),
    "pharmacy": (10, 150),
    "entertainment": (20, 250),
}

NUM_CARDS = 200
CARDS = [
    {"card_id": f"card_{i:04d}", "home_city": random.choice(CITIES)}
    for i in range(NUM_CARDS)
]

MERCHANT_SUFFIXES = ["Inc", "LLC", "Group", "Co", "Ltd", "Partners", "Holdings"]
MERCHANT_NAMES = [
    "Jackson", "Ryan", "Hudson", "Morris", "Stokes", "Bennett", "Carter",
    "Reed", "Foster", "Coleman", "Blake", "Hayes", "Nolan", "Grant",
]


def fake_company() -> str:
    return f"{random.choice(MERCHANT_NAMES)}-{random.choice(MERCHANT_NAMES)} {random.choice(MERCHANT_SUFFIXES)}"


_active_bursts: dict = {}


def make_legit_transaction(event_time: datetime) -> dict:
    card = random.choice(CARDS)
    category = random.choice(list(CATEGORIES.keys()))
    low, high = CATEGORIES[category]
    amount = round(random.uniform(low, high), 2)
    city = card["home_city"] if random.random() < 0.85 else random.choice(CITIES)
    return {
        "transaction_id": str(uuid.uuid4()),
        "card_id": card["card_id"],
        "amount": amount,
        "merchant": fake_company(),
        "category": category,
        "city": city,
        "timestamp": event_time.isoformat(),
        "is_fraud": False,
        "fraud_type": None,
    }


def make_fraud_transaction(event_time: datetime) -> dict:
    fraud_type = random.choice(["high_amount", "card_testing", "impossible_travel"])
    card = random.choice(CARDS)
    base = {
        "transaction_id": str(uuid.uuid4()),
        "card_id": card["card_id"],
        "merchant": fake_company(),
        "timestamp": event_time.isoformat(),
        "is_fraud": True,
        "fraud_type": fraud_type,
    }

    if fraud_type == "high_amount":
        category = random.choice(list(CATEGORIES.keys()))
        base.update({
            "amount": round(random.uniform(5000, 15000), 2),
            "category": category,
            "city": card["home_city"],
        })
    elif fraud_type == "card_testing":
        _active_bursts[card["card_id"]] = random.randint(5, 12)
        base.update({
            "amount": round(random.uniform(0.5, 5.0), 2),
            "category": "entertainment",
            "city": card["home_city"],
        })
    else:  # impossible_travel
        far_city = random.choice([c for c in CITIES if c != card["home_city"]])
        base.update({
            "amount": round(random.uniform(100, 1500), 2),
            "category": random.choice(list(CATEGORIES.keys())),
            "city": far_city,
        })
    return base


def make_burst_transaction(card_id: str, event_time: datetime) -> dict:
    return {
        "transaction_id": str(uuid.uuid4()),
        "card_id": card_id,
        "amount": round(random.uniform(0.5, 5.0), 2),
        "merchant": fake_company(),
        "category": "entertainment",
        "city": random.choice(CITIES),
        "timestamp": event_time.isoformat(),
        "is_fraud": True,
        "fraud_type": "card_testing",
    }


def next_transaction(event_time: datetime) -> dict:
    if _active_bursts:
        card_id = next(iter(_active_bursts))
        _active_bursts[card_id] -= 1
        if _active_bursts[card_id] <= 0:
            del _active_bursts[card_id]
        return make_burst_transaction(card_id, event_time)

    if random.random() < FRAUD_RATE:
        return make_fraud_transaction(event_time)
    return make_legit_transaction(event_time)


def main():
    print(f"Generating {NUM_TRANSACTIONS:,} transactions in batch...")

    # Spread timestamps across a simulated 30-day window, at roughly the
    # producer's original pace (~10 events/second), so the data looks like
    # a plausible historical slice rather than 150k events in one instant.
    start_time = datetime.now(timezone.utc) - timedelta(days=30)
    rows = []
    frauds = 0
    for i in range(NUM_TRANSACTIONS):
        event_time = start_time + timedelta(seconds=i / 10)
        tx = next_transaction(event_time)
        rows.append(tx)
        if tx["is_fraud"]:
            frauds += 1
        if (i + 1) % 25_000 == 0:
            print(f"  generated={i + 1:,}  frauds={frauds:,}  ({frauds / (i + 1):.2%})")

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Done. Wrote {len(df):,} rows to {OUTPUT_PATH}")
    print(f"Fraud count: {frauds:,} ({frauds / len(df):.3%})")
    print()
    print("Fraud type breakdown:")
    print(df[df["is_fraud"]]["fraud_type"].value_counts())


if __name__ == "__main__":
    main()
