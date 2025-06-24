"""Utility functions for preparing NVDA price and sentiment features."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.backends.backend_pdf import PdfPages
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _report(df: pd.DataFrame, title: str, n: int = 3) -> None:
    """Create console report for debugging purpose: shape, NaN count and preview."""
    print("\n" + "=" * 80)
    print(f"{title}  |  shape = {df.shape}")
    print("NaNs by column:\n", df.isna().sum().loc[lambda s: s.gt(0)].to_dict())
    print("Preview:\n", df.head(n))


def time_decay_impute(
    series: pd.Series, halflife: int = 3, fill_start: bool = True
) -> tuple[pd.Series, pd.Series]:
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


def load_price_data(price_path: str) -> pd.DataFrame:
    """Load NVDA price data and ensure numeric closing prices."""

    price = (
        pd.read_csv(price_path, parse_dates=["Date"], skiprows=[1])
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )
    price["Close"] = pd.to_numeric(price["Close"], errors="coerce")
    price["Close"].interpolate(method="linear", inplace=True)
    price["ln_close"] = np.log(price["Close"])
    return price


def load_sentiment_data(sentiment_path: str) -> pd.DataFrame:
    """Load already aggregated daily sentiment."""

    return (
        pd.read_csv(sentiment_path, parse_dates=["date"])
        .rename(columns={"date": "Date"})
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )


def merge_data(price: pd.DataFrame, sentiment: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Merge price and sentiment information on the Date index."""

    df = price.join(sentiment, how="left")
    df.index = pd.to_datetime(df.index)
    if debug:
        _report(price, "Before merge price")
        _report(sentiment, "Before merge sentiment")
        _report(df, "After merge")
    return df


def add_price_features(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Add returns, volatility, RSI and Bollinger bands.
    Log returns are defined as the difference of the log of closing prices.
    Simple returns are defined as the percentage change of closing prices.
    Volatility is defined as the rolling standard deviation of log returns.
    Bollinger bands are defined as the width of the bands around a 20-day moving average.
    RSI is defined as the 14-day relative strength index."""

    df['close_diff']= df["Close"].diff()
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()
    df["volatility"] = df["log_return"].rolling(7).std()
    df["rsi14"] = RSIIndicator(df["Close"], window=14).rsi()
    bb = BollingerBands(df["Close"], window=20)
    df["bb_width"] = bb.bollinger_wband()
    df["vol_roll5"] = df["log_return"].rolling(5).std()
    if debug:
        _report(df[["log_return", "rsi14", "bb_width", "vol_roll5"]], "After price features")
    return df


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = (1,3,5,10, 252), debug: bool = False) -> pd.DataFrame:
    """Create lagged versions of sentiment and return series."""

    for l in lags:
        df[f"log_return_l{l}"] = df["log_return"].shift(l)
        df[f"sentiment_l{l}"] = df["sentiment"].shift(l)
        df[f"close_l{l}"] = df["Close"].shift(l)
    if debug:
        lag_cols = df.filter(regex="_l(1|252)$").columns
        _report(df[lag_cols].tail(), "After lag features")
    return df


def add_calendar_features(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Encode day of week and month cycles."""

    df["dow"] = df.index.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    if debug:
        _report(df[["dow", "month_sin", "month_cos"]], "After calendar encodings")
    return df

def prepare_merged_data(
    price_path=os.path.join(SCRIPT_DIR, "..", "data", "nvda_stock.csv"),
    sentiment_path=os.path.join(SCRIPT_DIR, "..", "data", "nvda_sentiment_daily.csv"),
    output_path=os.path.join(SCRIPT_DIR, "..", "data", "nvda_merged.csv"),
    debug: bool = False,
) -> pd.DataFrame:
    """Create a feature rich, merged data set for downstream modelling."""

    price = load_price_data(price_path)
    sentiment = load_sentiment_data(sentiment_path)
    df = merge_data(price, sentiment, debug=debug)

    df["sentiment"], df["sentiment_missing"] = time_decay_impute(
        df["sentiment"], halflife=3
    )
    if debug:
        _report(df[["sentiment", "sentiment_missing"]], "After sentiment imputation")

    df = add_price_features(df, debug=debug)
    df = add_lag_features(df, debug=debug)
    df = add_calendar_features(df, debug=debug)

    df = df.dropna().sort_index()
    if debug:
        print("\nFinal data shape after dropna:", df.shape)
        print("Any remaining NaNs?", df.isna().any().any())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=True)
    print(f"Features saved to {output_path}")

    return df


def check_seasonality(series: pd.Series, period: int = 5) -> pd.DataFrame:
    """Return decomposition components of ``series``."""

    decomposition = seasonal_decompose(series.dropna(), model="multiplicative", period=period)

    return pd.DataFrame(
        {
            "seasonal": decomposition.seasonal,
            "trend": decomposition.trend,
            "residuals": decomposition.resid,
        }
    )


def check_stationarity(series: pd.Series) -> tuple[float, float]:
    """Return ADF and KPSS p-values for ``series``."""

    series = series.dropna()

    adf_test = adfuller(series, autolag="AIC")
    kpss_test = kpss(series, regression="c", nlags="auto", store=False)

    return adf_test[1], kpss_test[1]


def save_diagnostics_pdf(
    series: pd.Series,
    pdf_path: str,
    period: int = 5,
    lags: int = 3,
    model = "multiplicative",
) -> pd.DataFrame:
    """Create a PDF summarizing seasonality and stationarity diagnostics."""

    decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
    residuals = decomposition.resid.dropna()

    with PdfPages(pdf_path) as pdf:
        fig = decomposition.plot()
        fig.suptitle(f"{series.name} Decomposition", y=1.0)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(residuals, lags=lags, ax=axes[0])
        plot_pacf(residuals, lags=lags, ax=axes[1])
        axes[0].set_title("Residuals ACF")
        axes[1].set_title("Residuals PACF")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for data, title in [
            (series.dropna(), f"{series.name} Stationarity"),
            (residuals, "Residuals Stationarity"),
        ]:
            rolling_mean = data.rolling(30).mean()
            rolling_std = data.rolling(30).std()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data, label="Series", color="red")
            ax.plot(rolling_mean, label="Rolling Mean", color="blue")
            ax.plot(rolling_std, label="Rolling Std", color="green")
            ax.set_title(title)
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

            adf_p, kpss_p = check_stationarity(data)
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis("off")
            text = (
                f"{series.name} ADF p-value: {adf_p:.4f}\n"
                f"{series.name} KPSS p-value: {kpss_p:.4f}\n"
                "Interpretation:\n"
                f"ADF: {'Stationary' if adf_p < 0.05 else 'Non-stationary'}\n"
                f"KPSS: {'Non-stationary' if kpss_p < 0.05 else 'Stationary'}"
            )
            ax.text(0.1, 0.5, text, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

    return pd.DataFrame(
        {
            "seasonal": decomposition.seasonal,
            "trend": decomposition.trend,
            "residuals": decomposition.resid,
        }
    )


if __name__ == "__main__":
    price_file = os.path.join(SCRIPT_DIR, "..", "data", "nvda_stock.csv")
    sentiment_file = os.path.join(SCRIPT_DIR, "..", "data", "nvda_sentiment_daily.csv")

    price_df = load_price_data(price_file)
    pdf_file = os.path.join(SCRIPT_DIR, "..", "models", "diagnostics.pdf")

    # Since on raw closing prices no stationarity is detected,
    # it's better to use log returns for models which require stationary data
    # Build feature set and merge decomposition components
    df = prepare_merged_data(price_file, sentiment_file)
    decomposition = save_diagnostics_pdf(
        df["log_return"], pdf_file, model="additive")
    df = df.join(decomposition)
    print(df.head())