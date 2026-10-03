"""Check NADI 2024 / QADI parquet for labels and structure."""
import pandas as pd
import sys

print("=" * 60)
print("QADI 2024 parquet (D:/dacd2026/1_data/raw/qadi.parquet)")
print("=" * 60)
df = pd.read_parquet(r"D:/dacd2026/1_data/raw/qadi.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"dtypes:\n{df.dtypes}")
print()
print("First 5 rows:")
print(df.head())
print()
print("Label distribution:")
if "label" in df.columns:
    print(df["label"].value_counts().sort_index())
elif "dialect" in df.columns:
    print(df["dialect"].value_counts())
print()
if "label" in df.columns:
    n_classes = df["label"].nunique()
    print(f"Num classes: {n_classes}")
    if n_classes == 18:
        print("--> THIS IS THE 18-COUNTRY NADI 2024 FORMAT!")
print()

print("=" * 60)
print("amgadhasan parquet (D:/dacd2026/1_data/raw/amgadhasan.parquet)")
print("=" * 60)
df2 = pd.read_parquet(r"D:/dacd2026/1_data/raw/amgadhasan.parquet")
print(f"Shape: {df2.shape}")
print(f"Columns: {df2.columns.tolist()}")
print(f"First 5 rows:")
print(df2.head())
if "dialect" in df2.columns:
    print()
    print("Dialect distribution:")
    print(df2["dialect"].value_counts())
