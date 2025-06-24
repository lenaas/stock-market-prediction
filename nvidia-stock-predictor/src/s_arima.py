import os
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

# ARIMA/SARIMA
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Adjust SCRIPT_DIR
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load merged NVDA data with log_return and Close columns."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    # Ensure log_return and close lag exist
    df['log_return'] = df['log_return'].fillna(0)
    df['close_l1'] = df['Close'].shift(1)
    # Backfill to handle first row
    df[['close_l1', 'Close']] = df[['close_l1', 'Close']].fillna(method='bfill')
    return df


def train_arima(df: pd.DataFrame, seasonal: bool = False) -> None:
    """Train ARIMA/SARIMA on log_return, reconstruct Close price, and evaluate on prices."""
    series = df['log_return']
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    train_close_prev = df['close_l1'].iloc[:split]
    test_close = df['Close'].iloc[split:]
    test_close_prev = df['close_l1'].iloc[split:]

    # Auto ARIMA selection
    if seasonal:
        print('Finding best SARIMA model (seasonal=252)...')
        auto = pm.auto_arima(
            train,
            start_p=0, start_q=0, max_p=3, max_q=3,
            d=None, seasonal=True, m=252,
            start_P=0, start_Q=0, max_P=2, max_Q=2,
            D=None, trace=True,
            error_action='ignore', suppress_warnings=True, stepwise=True
        )
    else:
        print('Finding best ARIMA model...')
        auto = pm.auto_arima(
            train,
            start_p=0, start_q=0, max_p=3, max_q=3,
            d=None, seasonal=False,
            trace=True, error_action='ignore', suppress_warnings=True, stepwise=True
        )

    print(auto.summary())
    order = auto.order
    seasonal_order = auto.seasonal_order if seasonal else (0, 0, 0, 0)

    # Fit final model
    model = ARIMA(train, order=order, seasonal_order=seasonal_order)
    fitted = model.fit()

    # Forecast log returns
    n_periods = len(test)
    preds_log = fitted.forecast(steps=n_periods)

    # Reconstruct Close price forecasts
    preds_price = test_close_prev.values * np.exp(preds_log)

    # Compute metrics on prices
    mae_price = mean_absolute_error(test_close, preds_price)
    mse_price = mean_squared_error(test_close, preds_price)
    print(f"MAE (Close price): {mae_price:.4f}")
    print(f"MSE (Close price): {mse_price:.4f}")

    # Plot actual vs predicted Close prices
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test_close, label='Actual Close')
    plt.plot(test.index, preds_price, label='Predicted Close')
    title = 'SARIMA' if seasonal else 'ARIMA'
    plt.title(f"Actual vs Predicted Close Price ({title})")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    # Save figure
    fname = 'sarima_close_forecast.png' if seasonal else 'arima_close_forecast.png'
    out_path = os.path.join(SCRIPT_DIR, '..', 'models', fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.show()


def main():
    df = load_data()
    train_arima(df, seasonal=False)
    train_arima(df, seasonal=True)


if __name__ == '__main__':
    main()