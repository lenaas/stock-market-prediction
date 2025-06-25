import os
import pandas as pd
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import TimeSeriesSplit


# Adjust SCRIPT_DIR for interactive/notebook contexts
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load preprocessed NVDA data with log_return and Close columns."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    return df

def rolling_one_step_forecast(df: pd.DataFrame, features: List[str], window: int = 252):
    """
    Perform a rolling-window one-day-ahead forecast using Ridge regression with scaling.
    Returns:
      idxs   - list of dates forecasted (t+1)
      trues  - actual next-day close prices
      preds  - predicted next-day close prices
    """
    preds, trues, idxs = [], [], []

    for i in range(window, len(df) - 1):
        train = df.iloc[i - window : i]
        test  = df.iloc[i : i + 1]

        # Pipeline: scale → RidgeCV (alphas tuned via 5-fold CV)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-4, 4, 50), cv=TimeSeriesSplit(n_splits=5)))
        ])
        model.fit(train[features], train['target_return'])

        pred_log = model.predict(test[features])[0]
        close_t  = test['close_t'].iloc[0]

        preds.append(close_t * np.exp(pred_log))
        trues.append(test['true_close_t1'].iloc[0])

        # grab the actual next-day index for correct alignment
        idxs.append(df.index[i + 1])

    return idxs, trues, preds

def main():
    # 1) Load & sort
    df = load_data().sort_index()

    # 2) Create one-day-ahead target and base close
    df['target_return'] = df['log_return'].shift(-1)
    df['close_t']       = df['Close']
    df['true_close_t1'] = df['Close'].shift(-1)


    features = [
        'sentiment_l1', 'sentiment',
        'dow_sin', 'dow_cos',
        'month_sin','month_cos', 'high_vol'
    ]

    req_cols = features + ['target_return', 'close_t', 'true_close_t1']
    df_model = df.dropna(subset=req_cols)

    # 6) Perform rolling one-day-ahead forecast
    idxs, trues, preds = rolling_one_step_forecast(df_model, features, window=252)

    # 7) Compute evaluation metrics
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    print("Rolling One-Day-Ahead Forecast with RidgeCV + Scaling")
    print(f"→ MAE (next-day Close): {mae:.4f}")
    print(f"→ MSE: {mse:.4f}")
    print(f"→ RMSE: {rmse:.4f}")

    # Errors for plotting
    errors_abs = np.abs(np.array(preds) - np.array(trues))
    errors_sq  = errors_abs ** 2

    # 8) Plot actual vs predicted + errors (secondary y-axis)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(idxs, trues, label='Actual Close (t+1)')
    ax1.plot(idxs, preds, label='Predicted Close (t+1)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.scatter(idxs, errors_abs, label='Absolute Error', alpha=0.5, marker='x')
    ax2.set_ylabel('Absolute Error')
    ax2.legend(loc='upper right')

    plt.title('Rolling One-Step-Ahead: Actual vs Predicted Close (with Absolute Error)')
    plt.tight_layout()
    output_path = os.path.join(
        SCRIPT_DIR, '..', 'models',
        'linear_sentiment_v3_3.png'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

if __name__ == '__main__':
    main()

# Result:
#Rolling One-Day-Ahead Forecast with RidgeCV + Scaling (Features: sentiment_l1, sentiment, dow_sin, dow_cos, month_sin, month_cos)
#→ MAE (next-day Close): 2.6887
#→ MSE: 12.7004
#→ RMSE: 3.5638


# Rolling One-Day-Ahead Forecast with RidgeCV + Scaling
# Features: sentiment_l1, sentiment, dow_sin, dow_cos, month_sin, month_cos, high_vol
#→ MAE (next-day Close): 2.6863
#→ MSE: 12.6970
#→ RMSE: 3.5633
