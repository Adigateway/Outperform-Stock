import pandas as pd
import sqlite3
import numpy as np

############################################
# LOAD DATA
############################################

conn = sqlite3.connect("market_warehouse.db")

fund = pd.read_sql("SELECT * FROM fundamental_factors", conn)
price = pd.read_sql("SELECT * FROM price_factors", conn)

conn.close()

############################################
# DATE FORMATTING
############################################

fund["report_date"] = pd.to_datetime(fund["report_date"])
price["date"] = pd.to_datetime(price["date"])

############################################
# CONVERT DAILY → MONTHLY (ROBUST VERSION)
############################################

price = price.sort_values(["ticker", "date"])

# Set multi-index to avoid duplicate ticker insertion
price = price.set_index(["ticker", "date"])

monthly = (
    price
    .groupby(level=0)
    .resample("ME", level=1)
    .last()
    .reset_index()
)

############################################
# FORWARD 3-MONTH RETURN (LABEL)
############################################

monthly["fwd_3m_return"] = (
    monthly.groupby("ticker")["Close"]
    .shift(-3) / monthly["Close"] - 1
)

############################################
# MERGE FUNDAMENTALS
############################################

fund = fund.sort_values(["ticker", "report_date"])

merged = pd.merge_asof(
    monthly.sort_values("date"),
    fund.sort_values("report_date"),
    by="ticker",
    left_on="date",
    right_on="report_date",
    direction="backward"
)
merged["market_cap"] = merged["Close"] * merged["shares_outstanding"]
merged["log_market_cap"] = np.log(merged["market_cap"])
############################################
# CLEAN
############################################

merged = merged.dropna(subset=["fwd_3m_return"])

merged = merged.replace([np.inf, -np.inf], np.nan)

############################################
# SAVE
############################################

conn = sqlite3.connect("market_warehouse.db")

merged.to_sql(
    "model_dataset",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Model dataset built successfully.")
print("Rows:", len(merged))