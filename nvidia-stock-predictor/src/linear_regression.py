import os
import pandas as pd
from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Adjust SCRIPT_DIR if running interactively or from a notebook
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load merged NVDA data (with log_return and features)."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, "..", "data", "nvda_merged.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df


def train_linear_model(df: pd.DataFrame) -> LinearRegression:
    """Train regression on log returns and evaluate prediction error."""

    # Features for the model
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
    # Only need features and target for log-return modeling
    req_cols = features + [target]
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

    # Fit OLS regression on log returns
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Compute evaluation metrics on log-return predictions
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = model.score(X_test, y_test)

    print("\nLinear Regression on Log-Returns")
    print("MAE (log return):", round(mae, 6))
    print("MSE (log return):", round(mse, 6))
    print("R^2:", round(r2, 4))
    print("Coefficients:")
    for name, coef in zip(features, model.coef_):
        print(f"  {name}: {coef:.6f}")

    # Plot actual vs predicted log returns
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, y_test, label="Actual Log Return")
    plt.plot(test_df.index, preds, label="Predicted Log Return")
    plt.title("Actual vs Predicted Log Return")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(SCRIPT_DIR, "..", "models", "linear_log_return_forecast.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    return model


def main():
    df = load_data()
    train_linear_model(df)


if __name__ == "__main__":
    main()


# Results:
# Linear Regression on Log-Returns
# MAE (log return): 0.023593
# MSE (log return): 0.000912
# R^2: -0.1279
# Coefficients:
#   sentiment: 0.070683
#   volatility: 0.363273
#   rsi14: 0.001940
#   bb_width: -0.000966
#   vol_roll5: 0.306720
#   sentiment_l1: -0.002626
#   sentiment_l252: -0.000000
#   log_return_l1: -0.231224
#   log_return_l252: -0.089224