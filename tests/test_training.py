"""Training data generator and the training steps, on a small dataset."""

import generate_training_data as gen
import pandas as pd
import train_model
from conftest import tx


def test_fraud_patterns_follow_their_definitions(generated):
    fraud = generated[generated["is_fraud"]]
    high = fraud[fraud["fraud_type"] == "high_amount"]
    testing = fraud[fraud["fraud_type"] == "card_testing"]
    assert set(fraud["fraud_type"]) == {"high_amount", "card_testing", "impossible_travel"}
    assert high["amount"].min() >= 5000
    assert testing["amount"].max() <= 5.0
    assert (testing["category"] == "entertainment").all()
    assert generated["fraud_type"][~generated["is_fraud"]].isna().all()


def test_card_testing_starts_a_burst_on_the_same_card():
    gen._active_bursts.clear()
    gen._active_bursts["card_0001"] = 2
    first = gen.next_transaction(gen.START_TIME)
    second = gen.next_transaction(gen.START_TIME)
    assert first["card_id"] == second["card_id"] == "card_0001"
    assert first["fraud_type"] == "card_testing" and not gen._active_bursts


def test_home_city_comes_from_the_training_period_only():
    # 16 rows, so the 80% split puts the first 12 in training and the 4 Manaus rows in testing.
    # Over all rows A would be a Recife/Manaus tie, which resolves to Manaus.
    rows = [tx(f"C{i}", i) for i in range(8)]
    rows += [tx("A", 100 + i, "Recife") for i in range(4)]
    rows += [tx("A", 1000 + i, "Manaus") for i in range(4)]
    _, _, _, home = train_model.prepare(pd.DataFrame(rows))
    assert home["A"] == "Recife"


def test_train_produces_probabilities(generated):
    train_df, test_df, cols, _ = train_model.prepare(generated)
    model, weight = train_model.train(train_df, cols)
    assert weight > 1  # fraud is the minority class
    proba = model.predict_proba(test_df[cols])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
