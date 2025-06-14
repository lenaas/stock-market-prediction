import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def train_dual_forecaster():
    df = pd.read_csv("data/nvda_merged.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    close_series = df["Close"]
    sentiment_series = df["sentiment"].fillna(0)

    # Step 1: Train ARIMA
    arima_model = ARIMA(close_series, order=(2, 1, 2))
    arima_fitted = arima_model.fit()
    arima_pred = arima_fitted.predict(start=2, end=len(close_series)-1, typ='levels')

    # Step 2: Get residuals and align sentiment
    residuals = close_series[2:] - arima_pred[2:]
    sentiment_lag1 = sentiment_series.shift(1)[2:]

    # Train regression model on residuals
    sentiment_lag1 = sentiment_lag1.loc[residuals.index].fillna(0).values.reshape(-1, 1)
    residuals = residuals.values.reshape(-1, 1)

    reg = Ridge()
    reg.fit(sentiment_lag1, residuals)
    predicted_residuals = reg.predict(sentiment_lag1)

    # Step 3: Add correction to ARIMA forecast
    hybrid_forecast = arima_pred[2:] + predicted_residuals.flatten()

    # Evaluation
    actual = close_series[2:]
    mae = mean_absolute_error(actual, hybrid_forecast)
    mse = mean_squared_error(actual, hybrid_forecast)

    print(f"\n🔮 Dual-Forecaster Results — MAE: {mae:.2f}, MSE: {mse:.2f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"][2:], actual, label="Actual")
    plt.plot(df["Date"][2:], hybrid_forecast, label="Predicted (Dual-Forecaster)")
    plt.title("Dual-Forecaster: Actual vs Corrected Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/dual_forecaster_plot.png")
    plt.show()


if __name__ == "__main__":
    train_dual_forecaster()
