"""Shared test setup: make training/ importable and provide a small transaction set."""

import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "scoring"))

START = datetime(2026, 1, 1, tzinfo=UTC)


def tx(
    card: str,
    seconds: float,
    city: str = "Sao Paulo",
    category: str = "groceries",
    fraud_type: str | None = None,
    amount: float = 50.0,
) -> dict:
    """One transaction in the producer's format, `seconds` after START."""
    return {
        "transaction_id": f"{card}-{seconds}",
        "card_id": card,
        "amount": amount,
        "merchant": "Test Co",
        "category": category,
        "city": city,
        "timestamp": (START + timedelta(seconds=seconds)).isoformat(),
        "is_fraud": fraud_type is not None,
        "fraud_type": fraud_type,
    }


@pytest.fixture
def generated() -> pd.DataFrame:
    """3,000 transactions from the real generator, with its fraud patterns."""
    import generate_training_data as gen

    random.seed(7)
    gen._active_bursts.clear()
    rows = [gen.next_transaction(START + timedelta(seconds=i / 10)) for i in range(3000)]
    return pd.DataFrame(rows)
