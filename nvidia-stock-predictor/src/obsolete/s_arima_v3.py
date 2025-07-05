import os
import pandas as pd
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Adjust SCRIPT_DIR
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load NVDA data, set business-day frequency, backfill price lags."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    df = df.sort_index().asfreq('B')    # business‐day frequency
    df.index.freq = 'B'                 # ensure freq is set for forecasting
    # backfill any initial NaNs in close_l1 or Close
    df[['close_l1', 'Close']] = df[['close_l1', 'Close']].fillna(method='bfill')
    return df


def train_rolling_arima(
    df: pd.DataFrame,
    seasonal: bool = False,
    window: int = 252
):
    """
    Rolling one-day‐ahead ARIMAX/SARIMAX with sentiment exogenous.
    Auto‐selects orders via auto_arima, falls back if seasonal differencing fails.
    Reports price & return metrics.
    """
    # 1) Stationarity check on sentiment
    if 'sentiment' not in df.columns:
        raise KeyError("DataFrame must contain a 'sentiment' column")
    pval = adfuller(df['sentiment'].dropna())[1]
    if pval > 0.05:
        print(f"Sentiment ADF p={pval:.3f} → differencing once to enforce stationarity")
        df['sentiment'] = df['sentiment'].diff()

    # drop any NaNs before rolling
    df = df.dropna(subset=['log_return', 'sentiment', 'Close'])
    df.index = df.index.to_period('B')    # now every row is labeled with a B-period
    y_series     = df['log_return']
    price_series = df['Close']
    exog_series  = df['sentiment']

    # prepare first window for auto_arima
    y0 = y_series.iloc[:window]
    X0 = exog_series.iloc[:window].to_frame('sentiment')

    # 2) auto_arima with catch for seasonal differencing issue
    if seasonal:
        print("Finding best SARIMA model (m=252) with sentiment exog…")
        try:
            auto = pm.auto_arima(
                y0, X=X0,
                start_p=0, start_q=0, max_p=3, max_q=3,
                d=None,
                seasonal=True, m=252, D=None,
                stepwise=True,
                error_action='ignore', suppress_warnings=True
            )
        except ValueError as e:
            print("⚠️ Seasonal differencing error:", e)
            print("Falling back to non‐seasonal ARIMAX")
            seasonal = False
            auto = pm.auto_arima(
                y0, X=X0,
                start_p=0, start_q=0, max_p=3, max_q=3,
                d=None,
                seasonal=False,
                stepwise=True,
                error_action='ignore', suppress_warnings=True
            )
    else:
        print("Finding best ARIMAX model with sentiment exog…")
        auto = pm.auto_arima(
            y0, X=X0,
            start_p=0, start_q=0, max_p=3, max_q=3,
            d=None,
            seasonal=False,
            stepwise=True,
            error_action='ignore', suppress_warnings=True
        )

    print(auto.summary())
    order          = auto.order
    seasonal_order = auto.seasonal_order if seasonal else (0, 0, 0, 0)

    # 3) Rolling forecasts
    preds_price, trues_price, idxs = [], [], []
    preds_log,   trues_log   = [], []

    for i in range(window, len(y_series) - 1):
        y_train    = y_series.iloc[i - window : i]
        exog_train = exog_series.iloc[i - window : i].to_frame('sentiment')
        exog_test  = exog_series.iloc[i : i + 1].to_frame('sentiment')
        p_prev     = price_series.iloc[i]
        next_date  = y_series.index[i + 1]

        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False, method='lbfgs', maxiter=200)
        yhat_log = res.forecast(steps=1, exog=exog_test).iloc[0]

        preds_log.append(yhat_log)
        trues_log.append(y_series.iloc[i + 1])

        preds_price.append(p_prev * np.exp(yhat_log))
        trues_price.append(price_series.iloc[i + 1])
        idxs.append(next_date)

    # 4) Compute metrics
    mae_p  = mean_absolute_error(trues_price, preds_price)
    mse_p  = mean_squared_error(trues_price, preds_price)
    rmse_p = np.sqrt(mse_p)

    true_arr = np.array(trues_log, dtype=float)
    pred_arr = np.array(preds_log, dtype=float)
    mask     = np.isfinite(true_arr) & np.isfinite(pred_arr)
    true_arr = true_arr[mask]
    pred_arr = pred_arr[mask]

    mse_r          = mean_squared_error(true_arr, pred_arr)
    rmse_r         = np.sqrt(mse_r)
    baseline_mse_r = mean_squared_error(true_arr, np.zeros_like(true_arr))
    dir_acc_r      = (np.sign(pred_arr) == np.sign(true_arr)).mean()

    tag = 'SARIMAX' if seasonal else 'ARIMAX'
    print(f"\nRolling One-Day-Ahead {tag} (with sentiment exog)")
    print(f"→ Price MAE:                  {mae_p:.4f}")
    print(f"→ Price MSE:                  {mse_p:.4f}")
    print(f"→ Price RMSE:                 {rmse_p:.4f}")
    print(f"→ Return MSE:                 {mse_r:.6f}")
    print(f"→ Return RMSE:                {rmse_r:.6f}")
    print(f"→ Baseline Return MSE (zero): {baseline_mse_r:.6f}")
    print(f"→ Directional Accuracy (ret): {dir_acc_r:.3%}")

    # 5) Plot price‐level forecasts
    plt.figure(figsize=(10, 5))
    idxs = [p.to_timestamp() for p in idxs]
    plt.plot(idxs, trues_price, label='Actual Close (t+1)')
    plt.plot(idxs, preds_price, label='Predicted Close (t+1)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.title(f"{tag} w/ Sentiment Exog Forecast")
    plt.tight_layout()

    fname = f"rolling_{'sarimax' if seasonal else 'arimax'}_sentiment.png"
    out   = os.path.join(SCRIPT_DIR, '..', 'models', fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.show()

    return idxs, trues_price, preds_price


def main():
    df = load_data()
    train_rolling_arima(df, seasonal=False)  # ARIMAX
    train_rolling_arima(df, seasonal=True)   # SARIMAX


if __name__ == '__main__':
    main()



# Results without sentiment:
#Rolling One-Day-Ahead SARIMA
#→ MAE (next-day Close): 2.6518
#→ MSE: 12.5210
#→ RMSE: 3.5385

# Results with sentiment:
#Rolling One-Day-Ahead ARIMAX (with sentiment exog)
#→ Price MAE:                  2.8893
#→ Price MSE:                  14.4864
#→ Price RMSE:                 3.8061
#→ Return MSE:                 0.001113
#→ Return RMSE:                0.033356
#→ Baseline Return MSE (zero): 0.001083
#→ Directional Accuracy (ret): 44.248%