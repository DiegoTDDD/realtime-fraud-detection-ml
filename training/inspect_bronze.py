"""
Inspect the Bronze dataset from Project 1 (realtime-fraud-streaming).
Goal: confirm schema, row count, and the real fraud rate before using
this data to train the Project 6 model.
"""

import pandas as pd
import glob

BRONZE_PATH = r"C:\Users\Usuario\Desktop\realtime-fraud-streaming\data\bronze"

# Read all Bronze parquet files into a single DataFrame
files = glob.glob(f"{BRONZE_PATH}\\*.parquet")
print(f"Found {len(files)} Bronze parquet files.\n")

df = pd.read_parquet(BRONZE_PATH)

print("=== Schema ===")
print(df.dtypes)
print()

print("=== Row count ===")
print(f"Total rows: {len(df):,}")
print()

if "is_fraud" in df.columns:
    fraud_count = df["is_fraud"].sum()
    fraud_rate = fraud_count / len(df) * 100
    print("=== Fraud label distribution ===")
    print(f"Fraudulent transactions: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.3f}%")
else:
    print("WARNING: 'is_fraud' column not found. Columns available:")
    print(list(df.columns))

print()
print("=== Sample rows ===")
print(df.head(3).to_string())
