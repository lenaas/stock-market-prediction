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

    # chronological 80/20 split
    split = int(len(df_model) * 0.8)
    train_df, test_df = df_model.iloc[:split], df_model.iloc[split:]

    X_train, y_train = train_df[features], train_df["target_return_t+1"]
    X_test,  y_test  = test_df[features],  test_df["target_return_t+1"]

    # fit OLS
    lr = LinearRegression().fit(X_train, y_train)

    # forecast next-day log-returns
    preds_log = lr.predict(X_test)

    # rebuild next-day Close: Close_{t+1} = Close_t * exp(pred_return_{t+1})
    preds_close = test_df["close_l1"].values * np.exp(preds_log)
    true_close  = test_df["Close"].values          # this is Close_{t+1} thanks to dropna

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
    mse_r  = mean_squared_error(y_test, preds_log)
    rmse_r = np.sqrt(mse_r)
    baseline_mse = mean_squared_error(y_test, np.zeros_like(y_test))
    dir_acc = (np.sign(preds_log) == np.sign(y_test)).mean()

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



# Results
# Sentiment Only: MAE 2.8735, MSE 12.9655, R^2 0.9103, Coefficient sentiment -0.016435
# Sentiment + Lag 1 Sentiment: MAE 2.7662, MSE 12.7183, R^2 0.9120, Coefficient (sentiment) 0.026246
# Sentiment + Lagged Sentiments: MAE 3.1343, MSE 16.0328, R^2 0.8890, Coefficient (sentiment) 0.012388 
# Sentiment + Lag 3 Sentiments: MAE (Close price): 2.9469 MSE (Close price): 13.3302 R^2 (Close price): 0.9077 Coefficient (sentiment): -0.028210
# Sentiment + Lag 10 Sentiments: MAE (Close price): 3.1207 MSE (Close price): 15.1730 R^2 (Close price): 0.8950 Coefficient (sentiment): -0.040413
# Sentiment + Lag 252 Sentiment: MAE (Close price): 2.8735 MSE (Close price): 12.9655 R^2 (Close price): 0.9103 Coefficient (sentiment): -0.016435
# Best performed: Lag 1 and Lag 252 (the later on maybe only due to seasonal trend effects!)



#Linear Regression benchmark (same info set as ARIMAX)
#AR order used (lags): 1
#→ Price MAE:                  2.9186
#→ Price MSE:                  12.6712
#→ Price RMSE:                 3.5597
#→ Price MAPE:                 2.27%
#→ Price SMAPE:                2.27%
#→ Price R²:                   0.9092
#→ Return MSE:                 0.000755
#→ Return RMSE:                0.027478
#→ Baseline Return MSE (zero): 0.000730
#→ Directional Accuracy:       57.609%

#Coefficients:
#  sentiment_t     -0.023318
#  log_return_l1   -0.125171
#Intercept:         0.007514
