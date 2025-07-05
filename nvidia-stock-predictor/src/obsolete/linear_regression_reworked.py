"""
linear_regression_v1.py  ―  benchmark model that mirrors the ARIMAX
information set and one-day-ahead target.

It:
• Loads nvda_merged.csv
• Recreates close_l1 if it isn't in the CSV
• Builds sentiment_t + p lags of log_return, where p is the AR order
  chosen by pmdarima.auto_arima on the training window
• Trains an ordinary least-squares model
• Forecasts the next-day Close and prints the same evaluation metrics
  you use in the ARIMAX / SARIMAX script
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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
    if path is None:
        path = os.path.join(SCRIPT_DIR, "..", "data", "nvda_merged.csv")

    df = (
        pd.read_csv(path, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
        .asfreq("B")          # business-day calendar
    )

    # ensure the lagged price exists
    if "close_l1" not in df.columns:
        df["close_l1"] = df["Close"].shift(1)

    # back-fill the very first row (matches ARIMAX preprocessing)
    df[["close_l1", "Close"]] = df[["close_l1", "Close"]].fillna(method="bfill")
    return df


# ──────────────────────────────────────────────
# discover the AR order p that ARIMAX used
# ──────────────────────────────────────────────
def find_ar_order(series: pd.Series, exog: pd.Series) -> int:
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
    return max(p, 1)  # ensure at least 1 lag


# ──────────────────────────────────────────────
# linear regression using the *same* info set
# ──────────────────────────────────────────────
def train_lr_same_info(df: pd.DataFrame) -> Tuple[LinearRegression, List[str]]:
    df_feat = df.copy()
    df_feat["sentiment_t"] = df_feat["sentiment"]  # rename for clarity

    # discover p from the ARIMAX selection step
    ar_order = find_ar_order(df_feat["log_return"], df_feat["sentiment"])

    # build p lags of log_return
    lag_cols: List[str] = []
    for k in range(1, ar_order + 1):
        col = f"log_return_l{k}"
        df_feat[col] = df_feat["log_return"].shift(k)
        lag_cols.append(col)

    # one-day-ahead target
    df_feat["target_return_t+1"] = df_feat["log_return"].shift(-1)

    features = ["sentiment_t"] + lag_cols
    df_model = df_feat[features + ["target_return_t+1", "close_l1", "Close"]].dropna()

    # walk-forward validation using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    maes, mapes = [], []
    final_model = LinearRegression()
    for train_idx, test_idx in tscv.split(df_model):
        train_df = df_model.iloc[train_idx]
        test_df = df_model.iloc[test_idx]
        X_tr, y_tr = train_df[features], train_df["target_return_t+1"]
        X_te, y_te = test_df[features], test_df["target_return_t+1"]
        lr_fold = LinearRegression().fit(X_tr, y_tr)
        preds_fold = lr_fold.predict(X_te)
        preds_close_fold = test_df["close_l1"].values * np.exp(preds_fold)
        true_close_fold  = test_df["Close"].values
        maes.append(mean_absolute_error(true_close_fold, preds_close_fold))
        mapes.append(np.mean(np.abs((true_close_fold - preds_close_fold)/true_close_fold))*100)
        final_model = lr_fold

    lr = final_model

    # evaluate on the last split's test set
    preds_log = lr.predict(X_te)
    preds_close = test_df["close_l1"].values * np.exp(preds_log)
    true_close  = test_df["Close"].values

    # ── price-level metrics ───────────────────
    mae_p  = mean_absolute_error(true_close, preds_close)
    mse_p  = mean_squared_error(true_close, preds_close)
    rmse_p = np.sqrt(mse_p)
    mape_p = np.mean(np.abs((true_close - preds_close) / true_close)) * 100
    smape_p = (
        np.mean(2 * np.abs(true_close - preds_close) /
                (np.abs(true_close) + np.abs(preds_close))) * 100
    )
    r2_p   = r2_score(true_close, preds_close)

    # ── return-level metrics ──────────────────
    mse_r  = mean_squared_error(y_te, preds_log)
    rmse_r = np.sqrt(mse_r)
    baseline_mse = mean_squared_error(y_te, np.zeros_like(y_te))
    dir_acc = (np.sign(preds_log) == np.sign(y_te)).mean()

    # ── report ────────────────────────────────
    print("\nLinear Regression benchmark (same info set as ARIMAX)")
    print(f"AR order used (lags): {ar_order}")
    print(f"→ Price MAE:                  {mae_p:.4f}")
    print(f"→ Price MSE:                  {mse_p:.4f}")
    print(f"→ Price RMSE:                 {rmse_p:.4f}")
    print(f"→ Price MAPE:                 {mape_p:.2f}%")
    print(f"→ Price SMAPE:                {smape_p:.2f}%")
    print(f"→ Price R²:                   {r2_p:.4f}")
    print(f"→ Return MSE:                 {mse_r:.6f}")
    print(f"→ Return RMSE:                {rmse_r:.6f}")
    print(f"→ Baseline Return MSE (zero): {baseline_mse:.6f}")
    print(f"→ Directional Accuracy:       {dir_acc:.3%}")
    print("\nCoefficients:")
    for feat, coef in zip(features, lr.coef_):
        print(f"  {feat:<15} {coef: .6f}")
    print(f"Intercept:        {lr.intercept_: .6f}")
    print(f"\nCV MAE (avg): {np.mean(maes):.4f}")
    print(f"CV MAPE(avg): {np.mean(mapes):.2f}%")

    # ── quick diagnostic plot (optional) ──────
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, true_close, label="Actual Close")
    plt.plot(test_df.index, preds_close, label="Predicted Close")
    plt.title("Linear-Regression next-day Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return lr, features


# ──────────────────────────────────────────────
# main entry
# ──────────────────────────────────────────────
def main() -> None:
    df = load_data()
    train_lr_same_info(df)


if __name__ == "__main__":
    main()


#Linear Regression benchmark (same info set as ARIMAX)
#AR order used (lags): 1
#→ Price MAE:                  3.1626
#→ Price MSE:                  15.7407
#→ Price RMSE:                 3.9675
#→ Price MAPE:                 2.55%
#→ Price SMAPE:                2.54%
#→ Price R²:                   0.9046
#→ Return MSE:                 0.001093
#→ Return RMSE:                0.033053
#→ Baseline Return MSE (zero): 0.001064
#→ Directional Accuracy:       54.783%

#Coefficients:
#  sentiment_t     -0.003014
#  log_return_l1   -0.062598
#Intercept:         0.003351

#CV MAE (avg): 1.4410
#CV MAPE(avg): 2.42%
