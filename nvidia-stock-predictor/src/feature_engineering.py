import pandas as pd
import numpy as np
import os

def prepare_merged_data(
    price_path="data/nvda_stock.csv",
    sentiment_path="data/nvda_sentiment_daily.csv",
    output_path="data/nvda_merged.csv"
):
    # Load stock and sentiment data
    price_df = pd.read_csv(price_path, parse_dates=["Date"])
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])

    # Make sure Close column is numeric
    price_df["Close"] = pd.to_numeric(price_df["Close"], errors="coerce")

    # Merge on date
    merged = pd.merge(price_df, sentiment_df, left_on="Date", right_on="date", how="left")

    # Forward fill missing sentiment values
    merged["sentiment"] = merged["sentiment"].ffill()

    # Create return and lag features
    merged["return"] = merged["Close"].pct_change()
    merged["sentiment_lag1"] = merged["sentiment"].shift(1)
    merged["sentiment_lag2"] = merged["sentiment"].shift(2)
    merged["close_lag1"] = merged["Close"].shift(1)
    merged["close_lag2"] = merged["Close"].shift(2)

    # Drop rows with NaNs (due to lag/return shifts)
    merged.dropna(inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"✅ Saved merged features to {output_path}")
    return merged

if __name__ == "__main__":
    prepare_merged_data()
