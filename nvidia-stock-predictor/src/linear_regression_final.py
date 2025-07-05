"""
linear_regression_v1.py  ―  benchmark model that mirrors the ARIMAX
information set and one‑day‑ahead target (price *gap*), now enriched with
basic technical indicators.

Changes in this revision
------------------------
1. **RSI (14‑period)** is calculated and added as `rsi_t`.
2. **Simple moving averages (SMA)** at 5, 20, and 50 business‑day windows
   are calculated and added as `ma5_t`, `ma20_t`, `ma50_t`.
3. Feature list becomes
   `[sentiment_t, log_return_lags…, rsi_t, ma*_t]`.
4. All downstream logic (train/test split, metrics, plots) is unchanged.

Note: any *NaN* rows introduced by rolling calculations are dropped when
we build the modelling DataFrame so they don’t affect training.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import pmdarima as pm   # only used to discover AR order

# ──────────────────────────────────────────────
# paths & data loader
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the merged NVDA price/sentiment file and ensure required columns."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, "..", "data", "nvda_merged.csv")

    df = (
        pd.read_csv(path, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
        .asfreq("B")          # business‑day calendar
    )

    # ensure the lagged close exists (matches ARIMAX preprocessing)
    if "close_l1" not in df.columns:
        df["close_l1"] = df["Close"].shift(1)

    # back‑fill the very first row for Close and close_l1
    df[["close_l1", "Close"]] = df[["close_l1", "Close"]].fillna(method="bfill")

    return df


# ──────────────────────────────────────────────
# technical‑indicator helpers
# ──────────────────────────────────────────────

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder RSI implementation."""
    delta = close.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up   = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI and chosen SMAs in‑place and return the DataFrame."""
    df["rsi"] = compute_rsi(df["Close"], 14)
    for w in (5, 20, 50):
        df[f"ma{w}"] = df["Close"].rolling(w).mean()
    return df


# ──────────────────────────────────────────────
# discover the AR order p that ARIMAX used
# ──────────────────────────────────────────────

def find_ar_order(series: pd.Series, exog: pd.Series) -> int:
    """Run auto_arima on the training window to replicate ARIMAX lag order."""
    auto = pm.auto_arima(
        series.dropna(),
        X=exog.dropna().values.reshape(-1, 1),
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        d=None,
        seasonal=False,
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
    )
    p, _, _ = auto.order
    return max(p, 1)  # ensure at least one lag


# ──────────────────────────────────────────────
# linear regression using the *same* info set + tech indicators
# ──────────────────────────────────────────────

def train_lr_same_info(df: pd.DataFrame) -> Tuple[LinearRegression, List[str]]:
    """Train OLS on sentiment + AR lags + RSI + SMAs."""
    df_feat = df.copy()
    df_feat = add_technical_indicators(df_feat)
    df_feat["sentiment_t"] = df_feat["sentiment"]
    df_feat["sentiment_t-1"] = df_feat["sentiment"].shift(1)  # rename for clarity

    # discover p from the ARIMAX selection step
    ar_order = find_ar_order(df_feat["log_return"], df_feat["sentiment"])

    # build p lags of log_return
    lag_cols: List[str] = []
    for k in range(1, ar_order + 1):
        col = f"log_return_l{k}"
        df_feat[col] = df_feat["log_return"].shift(k)
        lag_cols.append(col)

    sentiment_cols = ["sentiment_t"]
    # tech‑indicator columns (no shift: they are information available at t)
    tech_cols = [
        "rsi",          # already suffixed _t conceptually
        "ma5",
        "ma20",
        "ma50",
    ]

    date_cols = ["dow_sin", "dow_cos", "month_sin", "month_cos"]

    # ── one‑day‑ahead *gap* target ───────────────────────────────────────
    df_feat["gap_t+1"] = df_feat["Close"].shift(-1) - df_feat["Open"].shift(-1)
    df_feat["open_t+1"] = df_feat["Open"].shift(-1)  # needed to rebuild price

    features =  lag_cols + tech_cols #+ sentiment_cols
    df_model = df_feat[features + ["gap_t+1", "open_t+1", "Close"]].dropna()

    # chronological 80/20 split
    split = int(len(df_model) * 0.8)
    train_df, test_df = df_model.iloc[:split], df_model.iloc[split:]

    X_train, y_train = train_df[features], train_df["gap_t+1"]
    X_test,  y_test  = test_df[features],  test_df["gap_t+1"]

    # fit OLS
    lr = LinearRegression().fit(X_train, y_train)

    # forecast next‑day gap (Close_{t+1} − Open_{t+1})
    preds_gap = lr.predict(X_test)

    # ── price‑level reconstruction ──────────────────────────────────────
    preds_close = test_df["open_t+1"].values + preds_gap
    true_close  = test_df["Close"].values  # equals Close_{t+1}

    # ── price‑level metrics ─────────────────────────────────────────────
    mae_p  = mean_absolute_error(true_close, preds_close)
    mse_p  = mean_squared_error(true_close, preds_close)
    rmse_p = np.sqrt(mse_p)
    mape_p = np.mean(np.abs((true_close - preds_close) / true_close)) * 100
    smape_p = (
        np.mean(2 * np.abs(true_close - preds_close) /
                (np.abs(true_close) + np.abs(preds_close))) * 100
    )
    r2_p   = r2_score(true_close, preds_close)

    # ── gap‑level metrics ───────────────────────────────────────────────
    mse_g  = mean_squared_error(y_test, preds_gap)
    rmse_g = np.sqrt(mse_g)
    baseline_mse = mean_squared_error(y_test, np.zeros_like(y_test))
    dir_acc = (np.sign(preds_gap) == np.sign(y_test)).mean()

        # DataFrame mit Zeitstempel, Features, Prediction und True Value
    results_df = test_df.copy()
    results_df["predicted_close"] = preds_close
    results_df["actual_close"] = true_close

    # Optional: auch die Gap-Vorhersage speichern
    results_df["predicted_gap"] = preds_gap
    results_df["actual_gap"] = y_test.values

    # Speichern als CSV
    results_df.to_csv(os.path.join(SCRIPT_DIR, "..", "data", "lr_predictions_without_sentiment.csv"))

    # ── report ──────────────────────────────────────────────────────────
    print("\nLinear Regression benchmark (same info set + tech indicators)")
    print(f"AR order used (lags): {ar_order}")
    print(f"→ Price MAE:                  {mae_p:.4f}")
    print(f"→ Price MSE:                  {mse_p:.4f}")
    print(f"→ Price RMSE:                 {rmse_p:.4f}")
    print(f"→ Price MAPE:                 {mape_p:.2f}%")
    print(f"→ Price SMAPE:                {smape_p:.2f}%")
    print(f"→ Price R²:                   {r2_p:.4f}")
    print(f"→ Gap  MSE:                   {mse_g:.6f}")
    print(f"→ Gap  RMSE:                  {rmse_g:.6f}")
    print(f"→ Baseline Gap MSE (zero):    {baseline_mse:.6f}")
    print(f"→ Directional Accuracy:       {dir_acc:.3%}")
    print("\nCoefficients:")
    for feat, coef in zip(features, lr.coef_):
        print(f"  {feat:<15} {coef: .6f}")
    print(f"Intercept:        {lr.intercept_: .6f}")

    # ── quick diagnostic plot (optional) ───────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, true_close, label="Actual Close")
    plt.plot(test_df.index, preds_close, label="Predicted Close")
    plt.title("Linear‑Regression next‑day Close  (tech inds)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("linear_regression_without_sentiment.png")

    return lr, features


# ──────────────────────────────────────────────
# main entry
# ──────────────────────────────────────────────

def main() -> None:
    df = load_data()
    train_lr_same_info(df)


if __name__ == "__main__":
    main()

#Linear Regression benchmark (same info set + tech indicators)
#AR order used (lags): 1
#→ Price MAE:                  1.8237
#→ Price MSE:                  7.1993
#→ Price RMSE:                 2.6832
#→ Price MAPE:                 1.50%
#→ Price SMAPE:                1.49%
#→ Price R²:                   0.9510
#→ Gap  MSE:                   11.368401
#→ Gap  RMSE:                  3.371706
#→ Baseline Gap MSE (zero):    11.312550
#→ Directional Accuracy:       50.000%

#Coefficients:
#  log_return_l1   -1.723689
#  rsi              0.004745
#  ma5             -0.002257
#  ma20            -0.027344
#  ma50             0.031028
#Intercept:        -0.201424

# Linear Regression benchmark (same info set + sentiment + tech indicators)
# AR order used (lags): 1
#→ Price MAE:                  1.7329
#→ Price MSE:                  6.3465
#→ Price RMSE:                 2.5192
#→ Price MAPE:                 1.42%
#→ Price SMAPE:                1.42%
#→ Price R²:                   0.9572
#→ Gap  MSE:                   10.741302
#→ Gap  RMSE:                  3.277393
#→ Baseline Gap MSE (zero):    10.664997
#→ Directional Accuracy:       48.462%

#Coefficients:
#  log_return_l1   -1.741718
#  rsi              0.003653
#  ma5              0.000002
#  ma20            -0.022438
#  ma50             0.022966
#  sentiment_t      0.068072
#Intercept:        -0.125432


# --------------------- with date cols -------------------
# Linear Regression benchmark (same info set + sentiment + tech indicators)
#AR order used (lags): 1
#→ Price MAE:                  1.7408
#→ Price MSE:                  6.3103
#→ Price RMSE:                 2.5120
#→ Price MAPE:                 1.42%
#→ Price SMAPE:                1.43%
#→ Price R²:                   0.9574
#→ Gap  MSE:                   10.667987
#→ Gap  RMSE:                  3.266189
#→ Baseline Gap MSE (zero):    10.664997
#→ Directional Accuracy:       51.538%

#Coefficients:
#  log_return_l1   -1.880116
#  rsi              0.001915
#  ma5              0.005754
#  ma20            -0.036634
#  ma50             0.032148
#  dow_sin         -0.150483
#  dow_cos          0.014250
#  month_sin        0.068260
#  month_cos        0.108125
#  sentiment_t     -0.205821
#Intercept:         0.026307

# Linear Regression benchmark (same info set + tech indicators)
#AR order used (lags): 1
#→ Price MAE:                  1.7327
#→ Price MSE:                  6.2930
#→ Price RMSE:                 2.5086
#→ Price MAPE:                 1.42%
#→ Price SMAPE:                1.42%
#→ Price R²:                   0.9575
#→ Gap  MSE:                   10.680280
#→ Gap  RMSE:                  3.268070
#→ Baseline Gap MSE (zero):    10.664997
#→ Directional Accuracy:       52.308%

#Coefficients:
#  log_return_l1   -1.870935
#  rsi              0.001952
#  ma5              0.006637
#  ma20            -0.036567
#  ma50             0.030804
#  dow_sin         -0.150527
#  dow_cos          0.014432
#  month_sin        0.075943
#  month_cos        0.095797
#Intercept:         0.026813