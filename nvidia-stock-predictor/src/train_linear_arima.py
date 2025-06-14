import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

def train_linear_model(df):
    features = ["close_lag1", "close_lag2", "sentiment_lag1", "sentiment_lag2"]
    target = "Close"

    X = df[features]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)

    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)

    print("\n📈 Linear Regression Results")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], y, label="Actual", alpha=0.7)
    plt.plot(df["Date"], preds, label="Predicted (Linear)", alpha=0.7)
    plt.title("Linear Regression: Actual vs Predicted NVDA Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/linear_regression_plot.png")
    plt.show()


def train_arima_model(df):
    series = df["Close"]

    model = ARIMA(series, order=(2, 1, 2))
    fitted = model.fit()

    # Get predictions with correct alignment
    preds = fitted.predict(start=2, end=len(series) - 1, typ="levels")

    # Align actual values to predicted ones
    actual = series[2:len(preds)+2]

    mae = mean_absolute_error(actual, preds)
    mse = mean_squared_error(actual, preds)

    print("\n🔁 ARIMA Results")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], series, label="Actual")
    plt.plot(df["Date"][2:len(preds)+2], preds, label="Predicted (ARIMA)")
    plt.title("ARIMA: Actual vs Predicted NVDA Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/arima_plot.png")
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("data/nvda_merged.csv", parse_dates=["Date"])
    train_linear_model(df)
    train_arima_model(df)
