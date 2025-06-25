import os
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Adjust SCRIPT_DIR
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load preprocessed NVDA data with sentiment, log_return, and Close columns."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    return df


def train_linear_model(df: pd.DataFrame) -> LinearRegression:
    """Train OLS on next-day log returns, reconstruct next-day Close, and evaluate."""
    # 1) define features
    features = ['sentiment_l1', 'sentiment', 'dow', 'month_sin']

    # 2) shift the target one day ahead
    df = df.copy()
    df['target_return'] = df['log_return'].shift(-1)
    # keep today's close so we can reconstruct tomorrow's
    df['close_t'] = df['Close']
    # actual next-day close for evaluation
    df['true_close_t1'] = df['Close'].shift(-1)

    # 3) drop any rows missing the features or targets
    req_cols = features + ['target_return', 'close_t', 'true_close_t1']
    df_model = df[req_cols].dropna()

    # 4) train-test split (80/20)
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df  = df_model.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df['target_return']
    X_test  = test_df[features]
    y_test  = test_df['target_return']

    # 5) fit OLS
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6) predict next-day log returns
    preds_log = model.predict(X_test)

    # 7) reconstruct next-day Close: Close_{t+1} = Close_t * exp(pred_log)
    preds_close = test_df['close_t'].values * np.exp(preds_log)
    true_close  = test_df['true_close_t1'].values

    # 8) metrics on the *next-day* price
    mae_price = mean_absolute_error(true_close, preds_close)
    mse_price = mean_squared_error(true_close, preds_close)
    r2_price  = r2_score(true_close, preds_close)

    print("\nLinear Regression One-Step-Ahead Forecast")
    print(f"MAE (next-day Close): {mae_price:.4f}")
    print(f"MSE (next-day Close): {mse_price:.4f}")
    print(f"R^2 (next-day Close): {r2_price:.4f}")
    print("Coefficients:")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat}: {coef:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")

    # 9) plot actual vs predicted next-day close
    # Errors for plotting
    errors_abs = np.abs(np.array(preds_close) - np.array(true_close))
    errors_sq  = errors_abs ** 2

    # 8) Plot actual vs predicted + errors (secondary y-axis)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(test_df.index, true_close, label='Actual Close (t+1)')
    ax1.plot(test_df.index, preds_close, label='Predicted Close (t+1)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.scatter(test_df.index, errors_abs, label='Absolute Error', alpha=0.5, marker='x')
    ax2.set_ylabel('Absolute Error')
    ax2.legend(loc='upper right')
    plt.show()
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, true_close,  label='Actual Close (t+1)')
    plt.plot(test_df.index, preds_close, label='Predicted Close (t+1)')
    plt.title('One-Step-Ahead Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        SCRIPT_DIR, '..', 'models',
        'linear_sentiment_l1_log_returnl3_onestep.png'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    return model


def main():
    df = load_data()
    train_linear_model(df)


if __name__ == '__main__':
    main()
