import pandas as pd
import sqlite3
import numpy as np

############################################
# LOAD
############################################

conn = sqlite3.connect("market_warehouse.db")
df = pd.read_sql("SELECT * FROM model_dataset", conn)
df["date"] = pd.to_datetime(df["date"])
conn.close()

############################################
# REMOVE MICROCAPS
############################################

df = df[df["market_cap"] > 300_000_000]  # > $300M
df = df[df["market_cap"].notna()]

############################################
# FACTOR COLUMNS
############################################

factor_cols = [
    "roa",
    "roe",
    "gross_margin",
    "operating_margin",
    "revenue_yoy",
    "income_yoy",
    "asset_growth",
    "debt_to_equity",
    "debt_to_assets",
    "current_ratio",
    "accrual_ratio",
    "log_market_cap",
]

############################################
# ENSURE FACTORS ARE NUMERIC
############################################

for col in factor_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

############################################
# WINSORIZE ONLY (1st–99th percentile)
# Clips extreme outliers but preserves the
# raw signal. HistGradientBoosting is tree-
# based so it does NOT need z-scoring —
# z-scoring here destroys cross-sectional
# signal before the model can use it.
############################################

for col in factor_cols:
    if col in df.columns:
        df[col] = df.groupby("date")[col].transform(
            lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
        )

############################################
# NOTE: z-score normalization removed.
# Cross-sectional z-scoring was collapsing
# all dates to mean=0, std=1, which made
# it impossible for the model to distinguish
# high-momentum from low-momentum periods.
# Sector-relative z-scores are computed
# inside build_ml_model.py instead, where
# they are applied correctly per sector×date.
############################################

############################################
# SAVE
############################################

conn = sqlite3.connect("market_warehouse.db")

df.to_sql(
    "model_dataset_clean",
    conn,
    if_exists="replace",
    index=False,
)

conn.close()

print("Clean dataset built.")
print("Rows:", len(df))