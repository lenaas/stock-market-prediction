from __future__ import annotations
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

script_dir = os.path.dirname(os.path.abspath(__file__))

def _report(df: pd.DataFrame, title: str, n: int = 3) -> None:
    """Create console report for debugging purpose: shape, NaN count and preview."""
    print("\n" + "=" * 80)
    print(f"{title}  |  shape = {df.shape}")
    print("NaNs by column:\n", df.isna().sum().loc[lambda s: s.gt(0)].to_dict())
    print("Preview:\n", df.head(n))


def time_decay_impute(series: pd.Series, halflife: int = 3, fill_start: bool = True) -> tuple[pd.Series, pd.Series]:
    """Impute missing sentiment values with an exponential weighted mean (past only).
    Using past only as the assumption is that only the past sentiment is influencing current sentiment.

    Parameters
    ----------
    series : pd.Series
        Sentiment series containing NaNs.
    halflife : int, default 3
        Half life for the exponential decay, expressed in periods (here: days).
        Set to 3 days since this should cover the typical sentiment decay period for news outlets.
        (For example: Social media would have < 12h decay, while news articles might last longer (this is at least our assumption).)

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (filled_series, missing_indicator)
    """
    # Compute exponential weighted mean with decay
    # Note: adjust=False means we only use past values, not future ones
    ewma = series.ewm(halflife=halflife, adjust=False).mean()
    filled = series.fillna(ewma)

    # zero-filling leading sentiments since we don't have past data here but we do need sth 
    if fill_start:                              
        filled = filled.fillna(0.0)
    # Create a missing indicator (1 for missing, 0 for filled) (could be useful for the models)
    missing = series.isna().astype(int)
    return filled, missing

def prepare_merged_data(
    price_path = os.path.join(script_dir, '..', 'data', 'nvda_stock.csv'),
    sentiment_path = os.path.join(script_dir, '..', 'data', 'nvda_sentiment_daily.csv'),
    output_path = os.path.join(script_dir, '..', 'data', 'nvda_merged.csv'),
    debug: bool = True,
) -> pd.DataFrame:
    """Create a feature rich, merged data set for downstream modelling.

    The function loads price and sentiment data, merges them on the date index,
    performs sentiment imputation (3 day half life), generates a variety of
    price based and calendar features, adds target columns, and persists the
    resulting data frame to *output_path*.
    """

    # 1) Load and basic cleaning
    price = (
        pd.read_csv(price_path, parse_dates=["Date"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )
    # Ensure the 'Close' column is numeric, coercing errors to NaN
    price["Close"] = pd.to_numeric(price["Close"], errors="coerce")

    # If any NaNs in Close column --> linear interpolation
    price["Close"].interpolate(method="linear", inplace=True)

    sentiment = (
        pd.read_csv(sentiment_path, parse_dates=["date"])
        .rename(columns={"date": "Date"})
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )
    if debug: _report(price, "Before merge price")
    if debug: _report(sentiment, "Before merge sentiment")


    df = price.join(sentiment, how="left")
    df.index = pd.to_datetime(df.index)

    if debug: _report(df, "After merge")

    # 2) Sentiment imputation (half‑life = 3 days)
    df["sentiment"], df["sentiment_missing"] = time_decay_impute(
        df["sentiment"], halflife=3
    )
    if debug: _report(df[["sentiment", "sentiment_missing"]], "After sentiment imputation")


    # 3) Price‑based features
    # Returns and volatility
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()
    df["volatility"] = df["log_return"].rolling(7).std()  # 7-day rolling volatility

    # RSI and Bollinger Bands
    # RSI (Relative Strength Index) with a 14-day window (default)
    df["rsi14"] = RSIIndicator(df["Close"], window=14).rsi()
    # Bollinger Bands with a 20-day window
    bb = BollingerBands(df["Close"], window=20)
    df["bb_width"] = bb.bollinger_wband()
    df["vol_roll5"] = df["log_return"].rolling(5).std()

    if debug: _report(df[["log_return", "rsi14", "bb_width", "vol_roll5"]], "After price features")


    # 4) Lag features (1–3 days)
    for l in (1, 2, 3):
        df[f"log_return_l{l}"] = df["log_return"].shift(l)
        df[f"sentiment_l{l}"] = df["sentiment"].shift(l)
    if debug: _report(df.filter(regex="_l[123]$"), "After lag features")

    # 5) Calendar encodings
    df["dow"] = df.index.dayofweek  # 0 = Monday
    # month_sin and month_cos for cyclical encoding of months
    # using cyclical encoding since otherwise december and january would be not treated as close
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    if debug: _report(df[["dow", "month_sin", "month_cos"]], "After calendar encodings")

    # 6) Targets
    df["target_return_1d"] = df["log_return"].shift(-1)  # next‑day return (regression)
    df["target_up"] = (df["target_return_1d"] > 0).astype(int)  # classification label

    if debug: _report(df[["target_return_1d", "target_up"]], "After target creation")

    # 7) Final clean‑up
    df = df.dropna().sort_index()

    if debug:
        print("\nFinal data shape after dropna:", df.shape)
        print("Any remaining NaNs?", df.isna().any().any())

    # 8) Persist feature set
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=True)
    print(f"Features saved to {output_path}")

    return df



def check_stationarity(series: pd.Series) -> float:
    """Run Augmented Dickey–Fuller test and return the p value.
    If p value < 0.05, the series is stationary.
    That means the null hypothesis of the test (that the series has a unit root) can be rejected.
    
    Consequences of non-stationarity:
    - Trends or seasonality can lead to misleading results in time series models.
    - Models that assume stationarity (like ARIMA) may not perform well.

    In case of non-stationarity, we need to difference the series or use transformations.
    """
    p_value = adfuller(series.dropna())[1]
    print(f"ADF p-value: {p_value:.3e}")
    return p_value


if __name__ == "__main__":
    # 1) Build feature set
    data = prepare_merged_data()

    # 2) Stationarity diagnostic on log returns
    _ = check_stationarity(data["log_return"])