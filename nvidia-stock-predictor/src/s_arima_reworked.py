import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Adjust SCRIPT_DIR for interactive/notebook contexts
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path=None):
    """Load NVDA data with log_return, Close, sentiment, and calendar features."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    # Ensure business‐day freq so auto_arima seasonal tests work
    df = df.sort_index().asfreq('B')
    df.index.freq = 'B'
    # backfill any initial NaNs
    df[['close_l1', 'Close']] = df[['close_l1', 'Close']].fillna(method='bfill')
    return df


def train_static_arimax(df, seasonal=False):
    """
    1) Splits first 80% for training, last 20% for testing
    2) ADF‐tests sentiment and differences once if needed
    3) auto_arima on train with exog = [sentiment, sentiment_l1, dow_sin, dow_cos, month_sin, month_cos]
    4) Fits one SARIMAX/ARIMAX on full train
    5) Forecasts all test points in one call
    6) Reports price‐ & return‐level metrics + directional accuracy
    """
    
    exog_vars = [
        'sentiment'
    ]
    # 1) Stationarity on sentiment
    pval = adfuller(df['sentiment'].dropna())[1]
    if pval > 0.05:
        print(f"Sentiment ADF p={pval:.3f}: differencing once")
        df['sentiment'] = df['sentiment'].diff()

    # 2) Drop all rows missing target or exog
    df = df.dropna(subset=['log_return', 'Close'] + exog_vars)

    # 3) Train/test split
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    y_train = train['log_return']
    y_test  = test['log_return']
    X_train = train[exog_vars]
    X_test  = test[exog_vars]

    # 4) auto_arima with exog
    print(f"\nAuto‐tuning {'SARIMA' if seasonal else 'ARIMA'} on {len(y_train)} points…")
    auto = pm.auto_arima(
        y_train, X = X_train,
        start_p=0, start_q=0, max_p=3, max_q=3,
        d=None,
        seasonal=seasonal, m=5 if seasonal else 1, D=1,
        stepwise=True, error_action='ignore', suppress_warnings=True
    )
    print(auto.summary())
    order          = auto.order
    seasonal_order = auto.seasonal_order if seasonal else (0, 0, 0, 0)

    # 5) Fit final SARIMAX/ARIMAX
    print(f"Fitting final {'SARIMAX' if seasonal else 'ARIMAX'} using seasonal_order={seasonal_order}…")
    model = SARIMAX(
        y_train,
        exog           = X_train,
        order          = order,
        seasonal_order = seasonal_order,
        enforce_stationarity  = True,
        enforce_invertibility = True
    )
    res = model.fit(disp=False, method='powell', maxiter=50, tol=1e-4 )

    # 6) Forecast log‐returns for entire test
    fc = res.get_forecast(steps=len(test), exog=X_test)
    yhat_log = fc.predicted_mean.values

    # 7) Reconstruct price forecasts
    close_prev = test['close_l1'].values  # price_{t}
    preds_price = close_prev * np.exp(yhat_log)
    trues_price = test['Close'].values

    # 8) Price‐level metrics
    mae_p  = mean_absolute_error(trues_price, preds_price)
    mse_p  = mean_squared_error(trues_price, preds_price)
    rmse_p = np.sqrt(mse_p)


    # 9) Return‐level metrics
    true_arr = test['log_return'].values
    pred_arr = yhat_log
    mse_r         = mean_squared_error(true_arr, pred_arr)
    rmse_r        = np.sqrt(mse_r)
    baseline_mse  = mean_squared_error(true_arr, np.zeros_like(true_arr))
    dir_acc       = (np.sign(pred_arr) == np.sign(true_arr)).mean()

    # ————————————
    # → Percent‐based price metrics

    # Mean Absolute Percentage Error
    mape_p = np.mean(
        np.abs((trues_price - preds_price) / trues_price)
    ) * 100

    # Symmetric Mean Absolute Percentage Error
    smape_p = np.mean(
        2 * np.abs(trues_price - preds_price) /
        (np.abs(trues_price) + np.abs(preds_price))
    ) * 100


    tag = 'SARIMAX' if seasonal else 'ARIMAX'
    print(f"\nStatic Train/Test {tag} Results")
    print(f"→ Price MAE:                  {mae_p:.4f}")
    print(f"→ Price MSE:                  {mse_p:.4f}")
    print(f"→ Price RMSE:                 {rmse_p:.4f}")
    print(f"→ Return MSE:                 {mse_r:.6f}")
    print(f"→ Return RMSE:                {rmse_r:.6f}")
    print(f"→ Baseline Return MSE (zero): {baseline_mse:.6f}")
    print(f"→ Directional Accuracy:       {dir_acc:.3%}")
    print(f"→ Price MAPE:  {mape_p:.2f}%")
    print(f"→ Price SMAPE: {smape_p:.2f}%")


    # 10) Plot price forecasts
    # plt.figure(figsize=(10,5))
    # plt.plot(test.index, trues_price, label='Actual Close')
    # plt.plot(test.index, preds_price, label='Predicted Close')
    # plt.xlabel('Date')
    # plt.ylabel('Close Price')
    # plt.title(f"{tag} Forecast (Test Set)")
    # plt.legend()
    # plt.tight_layout()

    errors_abs = np.abs(preds_price - trues_price)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(test.index, trues_price, label='Actual Close')
    ax1.plot(test.index, preds_price, label='Predicted Close')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.legend(loc='upper left')
    plt.title(f"{tag} Forecast (Test Set)")


    ax2 = ax1.twinx()
    ax2.scatter(test.index, errors_abs, label='Absolute Error', alpha=0.5, marker='x')
    ax2.set_ylabel('Absolute Error')
    ax2.legend(loc='upper right')

    out = os.path.join(
        SCRIPT_DIR, '..', 'models',
        f'static_{"sarimax" if seasonal else "arimax"}_sentiment.png'
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    plt.show()


def main():
    df = load_data()
    # try pure ARIMAX (no seasonal diff)
    train_static_arimax(df, seasonal=False)
    # then SARIMAX if you have >252 days in train
    train_static_arimax(df, seasonal=True)


if __name__ == '__main__':
    main()

#Static Train/Test SARIMAX Results
#→ Price MAE:                  3.7833
#→ Price MSE:                  24.0850
#→ Price RMSE:                 4.9076
##→ Return MSE:                 0.001547
#→ Return RMSE:                0.039332
#→ Baseline Return MSE (zero): 0.001022
#→ Directional Accuracy:       54.483%
#→ Price MAPE:  3.02%
#→ Price SMAPE: 3.01%