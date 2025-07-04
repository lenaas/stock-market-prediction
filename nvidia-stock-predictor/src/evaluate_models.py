import pandas as pd
import matplotlib.pyplot as plt

def load_predictions():
    # Paths to pre-saved prediction files or rebuild them from individual models if needed
    linear_df = pd.read_csv("data/nvda_merged.csv", parse_dates=["Date"])
    linear_df["Linear"] = pd.read_csv("models/linear_predictions.csv")["preds"]

    arima_df = pd.read_csv("models/arima_predictions.csv", parse_dates=["Date"])
    lstm_df = pd.read_csv("models/lstm_predictions.csv", parse_dates=["Date"])
    # dual_df = pd.read_csv("models/dual_forecaster_predictions.csv", parse_dates=["Date"])

    merged = linear_df[["Date", "Close", "Linear"]].copy()
    merged = merged.merge(arima_df[["Date", "ARIMA"]], on="Date", how="inner")
    merged = merged.merge(lstm_df[["Date", "LSTM"]], on="Date", how="inner")
    # merged = merged.merge(dual_df[["Date", "Dual"]], on="Date", how="inner")

    return merged


def plot_all_models(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Actual", linewidth=2)
    plt.plot(df["Date"], df["Linear"], label="Linear")
    plt.plot(df["Date"], df["ARIMA"], label="ARIMA")
    plt.plot(df["Date"], df["LSTM"], label="LSTM")
    plt.title("NVDA Stock: Actual vs All Models")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/comparison_plot.png")
    plt.show()


# def calculate_metrics(df):
#     from sklearn.metrics import mean_absolute_error, mean_squared_error

#     results = []
#     for col in ["Linear", "ARIMA", "LSTM", "Dual"]:
#         mae = mean_absolute_error(df["Close"], df[col])
#         mse = mean_squared_error(df["Close"], df[col])
#         results.append({"Model": col, "MAE": mae, "MSE": mse})

#     return pd.DataFrame(results)
def calculate_metrics(df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    results = []
    
    for col in ["Linear", "ARIMA", "LSTM"]:      # ← drop “Dual”
        if col not in df.columns:          # skip if the column is missing
            continue
        valid = df[["Close", col]].dropna()     # ← keep only rows with no NaN
        mae = mean_absolute_error(valid["Close"], valid[col])
        mse = mean_squared_error(valid["Close"], valid[col])
        results.append({"Model": col, "MAE": mae, "MSE": mse})

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = load_predictions()
    plot_all_models(df)
    metrics = calculate_metrics(df)
    print("\n📊 Model Performance Summary:")
    print(metrics.to_string(index=False))
    metrics.to_csv("models/metrics_summary.csv", index=False)
