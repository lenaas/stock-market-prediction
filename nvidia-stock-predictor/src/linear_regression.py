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
    """Load preprocessed NVDA data with sentiment, log_return, close_l1, and Close columns."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    return df


def train_linear_model(df: pd.DataFrame) -> LinearRegression:
    """Train simple OLS on log returns using only sentiment, reconstruct Close price, and evaluate."""
    # Use only sentiment as predictor
    features = ['sentiment']
    target = 'log_return'

    # Drop rows with missing sentiment or required columns
    df_model = df[features + [target, 'close_l1', 'Close']].dropna()

    # Train-test split (80/20)
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    prev_close = test_df['close_l1'].values
    true_close = test_df['Close'].values

    # Fit OLS regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict log returns
    preds_log = model.predict(X_test)

    # Reconstruct Close price forecasts: Close_t = Close_{t-1} * exp(log_return_t)
    preds_close = prev_close * np.exp(preds_log)

    # Compute evaluation metrics on prices
    mae_price = mean_absolute_error(true_close, preds_close)
    mse_price = mean_squared_error(true_close, preds_close)
    r2_price = r2_score(true_close, preds_close)

    print("\nLinear Regression (sentiment only)")
    print(f"MAE (Close price): {mae_price:.4f}")
    print(f"MSE (Close price): {mse_price:.4f}")
    print(f"R^2 (Close price): {r2_price:.4f}")
    print(f"Coefficient (sentiment): {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")

    # Plot actual vs predicted Close prices
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, true_close, label='Actual Close')
    plt.plot(test_df.index, preds_close, label='Predicted Close')
    plt.title('Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(SCRIPT_DIR, '..', 'models', 'linear_sentiment_only.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    return model


def main():
    df = load_data()
    train_linear_model(df)


if __name__ == '__main__':
    main()


# Results
# Sentiment Only: MAE 2.8735, MSE 12.9655, R^2 0.9103, Coefficient sentiment -0.016435
# Sentiment + Lag 1 Sentiment: MAE 2.7662, MSE 12.7183, R^2 0.9120, Coefficient (sentiment) 0.026246
# Sentiment + Lagged Sentiments: MAE 3.1343, MSE 16.0328, R^2 0.8890, Coefficient (sentiment) 0.012388 
# Sentiment + Lag 3 Sentiments: MAE (Close price): 2.9469 MSE (Close price): 13.3302 R^2 (Close price): 0.9077 Coefficient (sentiment): -0.028210
