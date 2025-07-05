import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Adjust SCRIPT_DIR for interactive/notebook contexts
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path=None):
    """Load NVDA data with log_return, Close, sentiment."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = (
        pd.read_csv(path, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
        .asfreq("B")
    )

    if "close_l1" not in df.columns:
        df["close_l1"] = df["Close"].shift(1)

    df[["close_l1", "Close"]] = df[["close_l1", "Close"]].fillna(method="bfill")
    return df

# ---------------------------------------------------------------------------
# technical indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = compute_rsi(df["Close"], 14)
    for w in (5, 20, 50):
        df[f"ma{w}"] = df["Close"].rolling(w).mean()
    return df


# ---------------------------------------------------------------------------
# SARIMAX training
# ---------------------------------------------------------------------------

def run_sarimax(df: pd.DataFrame, use_sentiment: bool) -> None:
    """Train/test SARIMAX with or without sentiment."""

    df_feat = add_technical_indicators(df.copy())
    df_feat["sentiment_t"] = df_feat["sentiment"]

    # next-day gap target and open for price reconstruction
    df_feat["gap_t+1"] = df_feat["Close"].shift(-1) - df_feat["Open"].shift(-1)
    df_feat["open_t+1"] = df_feat["Open"].shift(-1)

    tech_cols = ["rsi", "ma5", "ma20", "ma50"]
    exog_cols: List[str] = tech_cols + (["sentiment_t"] if use_sentiment else [])

    df_model = df_feat.dropna(subset=["gap_t+1", "open_t+1"] + exog_cols)

    split = int(len(df_model) * 0.8)
    train_df, test_df = df_model.iloc[:split], df_model.iloc[split:]

    y_train = train_df["gap_t+1"]
    y_test = test_df["gap_t+1"]
    X_train = train_df[exog_cols]
    X_test = test_df[exog_cols]

    print(
        f"Auto-tuning ARIMA order ({'with' if use_sentiment else 'without'} sentiment)..."
    )

    auto = pm.auto_arima(
        y_train,
        X=X_train,
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

    order = auto.order

    print(auto.summary())
    
    model = SARIMAX(
        y_train,
        exog           = X_train,
        order          = order,
        enforce_stationarity  = True,
        enforce_invertibility = True
    )
    res = model.fit(disp=False, method='powell', maxiter=50, tol=1e-4 )

    # 6) Forecast log‐returns for entire test
    fc = res.get_forecast(steps=len(X_test), exog=X_test)
    pred_gap = fc.predicted_mean.values

    preds_close = test_df["open_t+1"].values + pred_gap
    true_close = test_df["Close"].values

    mae_p = mean_absolute_error(true_close, preds_close)
    mse_p = mean_squared_error(true_close, preds_close)
    rmse_p = np.sqrt(mse_p)
    mape_p = np.mean(np.abs((true_close - preds_close) / true_close)) * 100
    smape_p = (
        np.mean(2 * np.abs(true_close - preds_close) /
                (np.abs(true_close) + np.abs(preds_close)))
        * 100
    )
    r2_p = r2_score(true_close, preds_close)

    mse_g = mean_squared_error(y_test, pred_gap)
    rmse_g = np.sqrt(mse_g)
    baseline_mse = mean_squared_error(y_test, np.zeros_like(y_test))
    dir_acc = (np.sign(pred_gap) == np.sign(y_test)).mean()

    tag = "with" if use_sentiment else "without"

    print(f"\nSARIMAX benchmark ({tag} sentiment)")
    print(f"ARIMA order used: {order}")
    print(f"→ Price MAE:                  {mae_p:.4f}")
    print(f"→ Price MSE:                  {mse_p:.4f}")
    print(f"→ Price RMSE:                 {rmse_p:.4f}")
    print(f"→ Price MAPE:                 {mape_p:.2f}%")
    print(f"→ Price SMAPE:                {smape_p:.2f}%")
    print(f"→ Price R²:                   {r2_p:.4f}")
    print(f"→ Gap  MSE:                   {mse_g:.6f}")
    print(f"→ Gap  RMSE:                  {rmse_g:.6f}")
    print(f"→ Baseline Gap MSE (zero):    {baseline_mse:.6f}")


    results_df = test_df.copy()
    results_df["predicted_close"] = preds_close
    results_df["actual_close"] = true_close
    results_df["predicted_gap"] = pred_gap
    results_df["actual_gap"] = y_test.values

    csv_name = f"sarima_predictions_{tag}_sentiment.csv"
    results_df.to_csv(os.path.join(SCRIPT_DIR, "..", "data", csv_name))

    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, true_close, label="Actual Close")
    plt.plot(test_df.index, preds_close, label="Predicted Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"SARIMAX next-day Close ({tag} sentiment)")
    plt.legend()
    plt.tight_layout()
    plot_name = f"sarima_{tag}_sentiment.png"
    plt.savefig(os.path.join(SCRIPT_DIR, "..", plot_name))
    plt.show()


def main():
    df = load_data()
    # try pure ARIMAX (no seasonal diff)
    run_sarimax(df, use_sentiment=True)
    run_sarimax(df, use_sentiment=False)


if __name__ == '__main__':
    main()

#SARIMAX benchmark (without sentiment)
#ARIMA order used: (1, 0, 1)
#→ Price MAE:                  1.8053
#→ Price MSE:                  7.0593
#→ Price RMSE:                 2.6569
#→ Price MAPE:                 1.48%
#→ Price SMAPE:                1.48%
#→ Price R²:                   0.9518
#→ Gap  MSE:                   10.942012
#→ Gap  RMSE:                  3.307871
#→ Baseline Gap MSE (zero):    10.938747

# SARIMAX benchmark (with sentiment)
#ARIMA order used: (1, 0, 1)
#→ Price MAE:                  1.7073
#→ Price MSE:                  6.1743
#→ Price RMSE:                 2.4848
#→ Price MAPE:                 1.40%
#→ Price SMAPE:                1.40%
#→ Price R²:                   0.9580
#→ Gap  MSE:                   10.223846
#→ Gap  RMSE:                  3.197475
#→ Baseline Gap MSE (zero):    10.237654
#Auto-tuning ARIMA order (without sentiment)...
#                               SARIMAX Results                                
#==============================================================================
#Dep. Variable:                      y   No. Observations:                  563
#Model:               SARIMAX(1, 0, 1)   Log Likelihood                -857.370
#Date:                Sat, 05 Jul 2025   AIC                           1728.739
#Time:                        15:48:22   BIC                           1759.072
#Sample:                             0   HQIC                          1740.581
#                                - 563                                         
#Covariance Type:                  opg                                         
#==============================================================================
#                 coef    std err          z      P>|z|      [0.025      0.975]
#------------------------------------------------------------------------------
#rsi            0.0013      0.001      1.201      0.230      -0.001       0.003
#ma5            0.0517      0.011      4.702      0.000       0.030       0.073
#ma20          -0.0866      0.014     -6.152      0.000      -0.114      -0.059
#ma50           0.0345      0.007      5.015      0.000       0.021       0.048
#ar.L1          0.7239      0.068     10.706      0.000       0.591       0.856
#ma.L1         -0.8701      0.061    -14.196      0.000      -0.990      -0.750
#sigma2         1.2240      0.039     31.467      0.000       1.148       1.300
#===================================================================================
#Ljung-Box (L1) (Q):                   0.60   Jarque-Bera (JB):              1334.91
#Prob(Q):                              0.44   Prob(JB):                         0.00
#Heteroskedasticity (H):               7.32   Skew:                            -0.83
#Prob(H) (two-sided):                  0.00   Kurtosis:                        10.36
#===================================================================================