import pandas as pd
import sqlite3
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

############################################
# SECTOR MAP
# Add / edit tickers here as your universe grows
############################################

SECTOR_MAP = {
    # Semiconductors
    "NVDA": "Semiconductors", "AMD": "Semiconductors", "INTC": "Semiconductors",
    "AVGO": "Semiconductors", "QCOM": "Semiconductors", "TXN": "Semiconductors",
    "MCHP": "Semiconductors", "NXPI": "Semiconductors", "ADI": "Semiconductors",
    "KLAC": "Semiconductors", "LRCX": "Semiconductors", "AMAT": "Semiconductors",
    "ON": "Semiconductors",   "SWKS": "Semiconductors", "TER": "Semiconductors",
    "MU": "Semiconductors",   "MPWR": "Semiconductors", "FSLR": "Semiconductors",
    "GLW": "Semiconductors",  "SMCI": "Semiconductors", "APH": "Semiconductors",
    "TEL": "Semiconductors",  "CIEN": "Semiconductors",
    # Software
    "MSFT": "Software", "ORCL": "Software", "CRM": "Software",  "ADBE": "Software",
    "NOW": "Software",  "INTU": "Software", "CDNS": "Software", "SNPS": "Software",
    "ADSK": "Software", "WDAY": "Software", "CRWD": "Software", "FTNT": "Software",
    "PANW": "Software", "DDOG": "Software", "PLTR": "Software", "VRSN": "Software",
    "FICO": "Software", "ROP": "Software",  "PTC": "Software",  "TYL": "Software",
    "GDDY": "Software", "GEN": "Software",  "APP": "Software",  "TRMB": "Software",
    "EPAM": "Software", "MSI": "Software",  "IT": "Software",   "ANET": "Software",
    # Hardware
    "AAPL": "Hardware", "DELL": "Hardware", "HPQ": "Hardware", "HPE": "Hardware",
    "STX": "Hardware",  "WDC": "Hardware",  "NTAP": "Hardware","JBL": "Hardware",
    "CDW": "Hardware",  "TDY": "Hardware",
    # IT Services
    "ACN": "IT Services", "IBM": "IT Services", "CSCO": "IT Services",
    "CTSH": "IT Services","FFIV": "IT Services","AKAM": "IT Services",
}

############################################
# FEATURES
############################################

FEATURE_COLS = [
    # Price / momentum
    "mom_3m", "mom_6m", "mom_12m", "vol_6m", "return",
    # Profitability
    "roa", "roe", "gross_margin", "operating_margin",
    # Leverage & liquidity
    "debt_to_equity", "debt_to_assets", "current_ratio",
    # Growth & quality
    "revenue_yoy", "income_yoy", "asset_growth", "accrual_ratio",
    # Size
    "log_market_cap",
]

############################################
# LOAD
############################################

conn = sqlite3.connect("market_warehouse.db")
df = pd.read_sql("SELECT * FROM model_dataset_clean ORDER BY date, ticker", conn)
conn.close()

df["date"] = pd.to_datetime(df["date"])
print(f"Loaded {len(df):,} rows  |  {df['ticker'].nunique()} tickers  |  {df['date'].nunique()} dates")

############################################
# ATTACH SECTORS
############################################

df["sector"] = df["ticker"].map(SECTOR_MAP)

unmapped = df[df["sector"].isna()]["ticker"].unique()
if len(unmapped):
    print(f"WARNING: {len(unmapped)} tickers have no sector mapping → {list(unmapped)}")

############################################
# TARGET: outperform sector median next 3 months
############################################

sector_median = df.groupby(["date", "sector"])["fwd_3m_return"].transform("median")
df["outperforms"] = (df["fwd_3m_return"] > sector_median).astype(int)

print(f"Overall outperformance rate: {df['outperforms'].mean():.1%}  (expect ~50%)")

############################################
# SECTOR-RELATIVE FEATURES
# Z-scored within sector×date so the model
# sees each stock's rank vs its peers.
# Done here (not in clean_model_dataset.py)
# so the raw signal is preserved upstream.
############################################

REL_COLS = ["mom_3m", "mom_6m", "mom_12m", "vol_6m", "log_market_cap", "roa", "roe"]

for col in REL_COLS:
    new_col = f"{col}_rel"
    df[new_col] = df.groupby(["date", "sector"])[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    FEATURE_COLS.append(new_col)

############################################
# PREPARE ARRAYS
# HistGradientBoostingClassifier handles NaNs
# natively — no dropna needed, no rows lost.
############################################

df_model = df.dropna(subset=["outperforms", "fwd_3m_return"]).copy()
df_model = df_model.sort_values("date").reset_index(drop=True)

print(f"Rows for modelling: {len(df_model):,}  (NaNs in features are OK — handled by model)")
print(f"Date range: {df_model['date'].min().date()}  →  {df_model['date'].max().date()}")
print(f"Features: {len(FEATURE_COLS)}")

X = df_model[FEATURE_COLS].values
y = df_model["outperforms"].values

############################################
# TIME-SERIES CROSS VALIDATION
############################################

print("\n── Time-Series Cross Validation ──────────────────────────")

tscv = TimeSeriesSplit(n_splits=5)
cv_aucs = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    cv_aucs.append(auc)

    print(f"  Fold {fold + 1}  |  AUC = {auc:.4f}  |  n_test = {len(y_test)}")

print(f"\n  Mean AUC: {np.mean(cv_aucs):.4f}  ±  {np.std(cv_aucs):.4f}")
print("────────────────────────────────────────────────────────────")

############################################
# TRAIN FINAL MODEL ON ALL DATA
############################################

final_model = HistGradientBoostingClassifier(
    max_iter=300,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42,
)
final_model.fit(X, y)

print("\nFinal model trained on all data.")

############################################
# FEATURE IMPORTANCE
# HistGradientBoosting doesn't expose
# feature_importances_ directly — use
# permutation importance on the last CV
# fold as a proxy (fast, no data leakage).
############################################

from sklearn.inspection import permutation_importance

# Use last fold's test set for permutation importance
last_train, last_test = list(tscv.split(X))[-1]
perm = permutation_importance(
    final_model, X[last_test], y[last_test],
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": perm.importances_mean,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\n── Top 10 Features (permutation importance) ────────────────")
print(importance_df.head(10).to_string(index=False))

############################################
# PREDICTIONS ON LATEST SNAPSHOT DATE
############################################

latest_date = df_model["date"].max()
latest      = df_model[df_model["date"] == latest_date].copy()
X_latest    = latest[FEATURE_COLS].values
latest["outperform_prob"] = final_model.predict_proba(X_latest)[:, 1]

print(f"\n── Predictions for {latest_date.date()} ─────────────────────────────")

for sector in sorted(latest["sector"].dropna().unique()):
    top5 = (
        latest[latest["sector"] == sector]
        .sort_values("outperform_prob", ascending=False)
        .head(5)[["ticker", "outperform_prob", "mom_3m", "roa"]]
    )
    print(f"\n  {sector}")
    print(f"  {'Ticker':<8} {'Prob':>6}  {'Mom3M':>7}  {'ROA':>6}")
    print(f"  {'-'*36}")
    for _, row in top5.iterrows():
        print(
            f"  {row['ticker']:<8} "
            f"{row['outperform_prob']:>6.1%}  "
            f"{row['mom_3m']:>+7.1%}  "
            f"{row['roa']:>6.2f}"
        )

############################################
# SAVE PREDICTIONS TO DATABASE
############################################

out_cols = [
    "ticker", "sector", "date", "outperform_prob",
    "mom_3m", "mom_6m", "roa", "roe",
    "revenue_yoy", "gross_margin", "market_cap",
]
out_cols = [c for c in out_cols if c in latest.columns]

conn = sqlite3.connect("market_warehouse.db")
latest[out_cols].to_sql("ml_predictions", conn, if_exists="replace", index=False)
conn.close()

print("\n✓ Predictions saved to table: ml_predictions")

############################################
# SAVE MODEL TO DISK
# Note: no scaler needed — HistGradientBoosting
# is tree-based so feature scaling has no effect.
############################################

with open("model.pkl", "wb") as f:
    pickle.dump({
        "model":    final_model,
        "features": FEATURE_COLS,
    }, f)

print("✓ Model saved to: model.pkl")

############################################
# SAVE CV RESULTS
############################################

with open("cv_results.json", "w") as f:
    json.dump({
        "fold_aucs":  cv_aucs,
        "mean_auc":   float(np.mean(cv_aucs)),
        "std_auc":    float(np.std(cv_aucs)),
        "n_features": len(FEATURE_COLS),
        "n_rows":     len(df_model),
        "latest_date": str(latest_date.date()),
    }, f, indent=2)

print("✓ CV results saved to: cv_results.json")
print("\nDone.")