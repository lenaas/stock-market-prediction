import os
import pandas as pd
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load merged NVDA data (with decomposition columns)."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, "..", "data", "nvda_merged.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df


def train_linear_model(df: pd.DataFrame) -> LinearRegression:
    """Train regression on residuals of log-returns and reconstruct close price."""

    # Features for the model (include commas between all entries!)
    features = [
        "sentiment",
        "volatility",
        "rsi14",
        "bb_width",
        "vol_roll5",
        "sentiment_l1",
        "sentiment_l252",
        "log_return_l1",
        "log_return_l252",
    ]

    target = "log_return"
    # Required columns: features + target + decomposition + lagged close + actual close
    req_cols = features + [target, "close_l1", "Close"]
    df[req_cols] = df[req_cols].fillna(0)

    # Train-test split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Prepare data
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Fit OLS regression on residuals
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds_residual = model.predict(X_test)

    # Reconstruct predicted log-return: add back trend + seasonal
    predicted_log_return = preds_residual + test_df["trend"] + test_df["seasonal"]
    # Convert log-return to price: Close_t = Close_{t-1} * exp(log_return_t)
    predicted_close = test_df["close_l1"] * np.exp(predicted_log_return)

    # Metrics on predicted close price
    mae = mean_absolute_error(test_df["Close"], predicted_close)
    mse = mean_squared_error(test_df["Close"], predicted_close)
    r2 = model.score(X_test, y_test)

    print("\nLinear Regression on Residuals")
    print("MAE (predicted close):", round(mae, 4))
    print("MSE (predicted close):", round(mse, 4))
    print("R^2 on residual regression:", round(r2, 4))
    print("Coefficients:")
    for name, coef in zip(features, model.coef_):
        print(f"  {name}: {coef:.4f}")

    # Plot actual vs predicted close price
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, test_df["Close"], label="Actual")
    plt.plot(test_df.index, predicted_close, label="Predicted")
    plt.title("Actual vs Predicted Close Price (Residual Regression)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(SCRIPT_DIR, "..", "models", "linear_regression_forecast.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    return model


def main():
    df = load_data()
    train_linear_model(df)


if __name__ == "__main__":
    main()
