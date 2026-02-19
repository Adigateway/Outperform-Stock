import pandas as pd
import sqlite3

############################################
# LOAD FINANCIALS
############################################

conn = sqlite3.connect("market_warehouse.db")
df = pd.read_sql("SELECT * FROM financials", conn)
conn.close()

df["report_date"] = pd.to_datetime(df["report_date"])
df = df.sort_values(["ticker", "report_date"])

############################################
# PROFITABILITY FACTORS
############################################

df["roa"] = df["net_income"] / df["total_assets"]
df["roe"] = df["net_income"] / df["total_equity"]
df["gross_margin"] = df["gross_profit"] / df["revenue"]
df["operating_margin"] = df["operating_income"] / df["revenue"]

############################################
# LEVERAGE FACTORS
############################################

df["debt_to_equity"] = df["Total Debt"] / df["total_equity"]
df["debt_to_assets"] = df["Total Debt"] / df["total_assets"]

############################################
# LIQUIDITY
############################################

df["current_ratio"] = df["current_assets"] / df["current_liabilities"]

############################################
# GROWTH FACTORS (YoY)
############################################

df["revenue_yoy"] = df.groupby("ticker")["revenue"].pct_change(4)
df["income_yoy"] = df.groupby("ticker")["net_income"].pct_change(4)
df["asset_growth"] = df.groupby("ticker")["total_assets"].pct_change(4)

############################################
# QUALITY
############################################

df["accrual_ratio"] = (
    (df["net_income"] - df["operating_cashflow"]) 
    / df["total_assets"]
)

############################################
# CLEAN
############################################

df = df.replace([float("inf"), -float("inf")], None)

############################################
# SAVE
############################################

conn = sqlite3.connect("market_warehouse.db")

df.to_sql(
    "fundamental_factors",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Fundamental factors built successfully.")