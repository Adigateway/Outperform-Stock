import pandas as pd
import sqlite3

############################################
# LOAD SIMFIN CSV FILES
############################################
income = pd.read_csv("/Users/adirajuadityasrivatsa/Documents/M/Stock Research/Version 8/Datasets/us-income-quarterly.csv", sep=';')
balance = pd.read_csv("/Users/adirajuadityasrivatsa/Documents/M/Stock Research/Version 8/Datasets/us-balance-quarterly.csv", sep=';')
cashflow = pd.read_csv("/Users/adirajuadityasrivatsa/Documents/M/Stock Research/Version 8/Datasets/us-cashflow-quarterly.csv", sep=';')

print("Income rows:", len(income))
print("Balance rows:", len(balance))
print("Cashflow rows:", len(cashflow))

############################################
# SELECT ONLY IMPORTANT COLUMNS
############################################

income = income[[
    "Ticker",
    "Report Date",
    "Revenue",
    "Gross Profit",
    "Operating Income (Loss)",
    "Net Income",
    "Research & Development",
    "Selling, General & Administrative",
    "Shares (Basic)"
]]

balance = balance[[
    "Ticker",
    "Report Date",
    "Total Assets",
    "Total Current Assets",
    "Total Current Liabilities",
    "Total Equity",
    "Short Term Debt",
    "Long Term Debt"
]]

cashflow = cashflow[[
    "Ticker",
    "Report Date",
    "Net Cash from Operating Activities",
    "Depreciation & Amortization"
]]

############################################
# CREATE TOTAL DEBT
############################################

balance["Total Debt"] = (
    balance["Short Term Debt"].fillna(0) +
    balance["Long Term Debt"].fillna(0)
)

############################################
# MERGE EVERYTHING
############################################

df = income.merge(balance, on=["Ticker", "Report Date"], how="left")
df = df.merge(cashflow, on=["Ticker", "Report Date"], how="left")

############################################
# CLEAN COLUMN NAMES
############################################

df = df.rename(columns={
    "Ticker": "ticker",
    "Report Date": "report_date",
    "Revenue": "revenue",
    "Gross Profit": "gross_profit",
    "Operating Income (Loss)": "operating_income",
    "Net Income": "net_income",
    "Research & Development": "rd_expense",
    "Selling, General & Administrative": "sga_expense",
    "Total Assets": "total_assets",
    "Total Current Assets": "current_assets",
    "Total Current Liabilities": "current_liabilities",
    "Total Equity": "total_equity",
    "Net Cash from Operating Activities": "operating_cashflow",
    "Depreciation & Amortization": "depreciation",
    "Shares (Basic)": "shares_outstanding",
})

df["report_date"] = pd.to_datetime(df["report_date"])

############################################
# DROP EMPTY
############################################

df = df.dropna(subset=["revenue", "net_income"])

print("Final financial rows:", len(df))

############################################
# SAVE TO DATABASE
############################################

conn = sqlite3.connect("market_warehouse.db", timeout=30)

df.to_sql(
    "financials",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Financials table rebuilt successfully.")