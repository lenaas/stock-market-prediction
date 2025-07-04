#!/usr/bin/env python3
"""
feature_engineering.py

Utility functions & a CLI to prepare NVDA price + sentiment features,
merge them into a single DataFrame, and optionally emit diagnostics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


# def load_price_data(path: Path) -> pd.DataFrame:
#     """Load NVDA price CSV, ensure numeric Close, interpolate missing, compute ln_close."""
#     if not path.exists():
#         raise FileNotFoundError(f"Price file not found: {path}")
#     df = (
#         pd.read_csv(path, parse_dates=["Date"])
#         .drop_duplicates(subset="Date")
#         .set_index("Date")
#         .sort_index()
#     )
    
#     df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
#     df["Close"].interpolate(method="time", inplace=True)
#     df["Close"].fillna(method="bfill", inplace=True)
#     df["ln_close"] = np.log(df["Close"])
#     return df

def load_price_data(path: Path) -> pd.DataFrame:
    """Load NVDA price CSV, ensure numeric Close, interpolate missing, compute ln_close."""
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    # read → deduplicate → set Date index
    df = (
        pd.read_csv(path, parse_dates=["Date"])
          .drop_duplicates(subset="Date")
          .set_index("Date")
          .sort_index()
    )

    # ── NEW: clean the index so time-based interpolation doesn’t choke ──
    df = df.loc[~df.index.isna()]                     # drop any NaT rows
    df = df.loc[~df.index.duplicated(keep="first")]   # drop duplicate dates
    # -------------------------------------------------------------------

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Close"].interpolate(method="time", inplace=True)
    df["Close"].fillna(method="bfill", inplace=True)

    df["ln_close"] = np.log(df["Close"])
    return df


def load_sentiment_data(path: Path) -> pd.DataFrame:
    """Load daily sentiment CSV (date, sentiment)."""
    if not path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {path}")
    df = (
        pd.read_csv(path, parse_dates=["date"])
        .rename(columns={"date": "Date"})
        .drop_duplicates(subset="Date")
        .set_index("Date")
        .sort_index()
    )
    return df[["sentiment"]]


def time_decay_impute(
    series: pd.Series, halflife: int = 3, fill_start: bool = True
) -> tuple[pd.Series, pd.Series]:
    """EWMA‐impute NaNs in `series`, return (filled, was_missing_flag)."""
    ewma = series.ewm(halflife=halflife, adjust=False).mean()
    filled = series.fillna(ewma)
    if fill_start:
        filled = filled.fillna(0.0)
    missing_flag = series.isna().astype(int)
    return filled, missing_flag


def merge_data(
    price: pd.DataFrame, sentiment: pd.DataFrame, debug: bool = False
) -> pd.DataFrame:
    """Join price & sentiment on Date index (left‐join)."""
    df = price.join(sentiment, how="left")
    if debug:
        logging.debug("After merge, head:\n%s", df.head(3))
        logging.debug("NaNs per column:\n%s", df.isna().sum())
    return df


def add_price_features(
    df: pd.DataFrame,
    return_window: int = 7,
    rsi_window: int = 14,
    bb_window: int = 20,
    debug: bool = False,
) -> pd.DataFrame:
    """Compute returns, volatility, RSI, Bollinger‐width, etc."""
    df = df.copy()
    df["simple_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()
    df["volatility"] = df["log_return"].rolling(return_window, min_periods=1).std()
    df["high_vol"] = (df["volatility"] > df["volatility"].median()).astype(int)
    df["return_ma3"] = df["log_return"].rolling(3, min_periods=1).sum()

    df["rsi"] = RSIIndicator(df["Close"], window=rsi_window).rsi()
    bb = BollingerBands(df["Close"], window=bb_window)
    df["bb_width"] = bb.bollinger_wband()
    df["vol_roll5"] = df["log_return"].rolling(5, min_periods=1).std()

    if debug:
        logging.debug("Price features head:\n%s", df[["log_return", "volatility", "rsi", "bb_width"]].head(3))
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int],
    debug: bool = False,
) -> pd.DataFrame:
    """Add lagged versions of log_return, sentiment, and Close."""
    df = df.copy()
    for lag in lags:
        df[f"log_return_l{lag}"] = df["log_return"].shift(lag)
        df[f"sentiment_l{lag}"] = df["sentiment"].shift(lag)
        df[f"close_l{lag}"] = df["Close"].shift(lag)
    df["return_accel"] = df["log_return"] - df["log_return_l1"]
    if debug:
        cols = [f"log_return_l{lags[0]}", f"sentiment_l{lags[0]}", f"close_l{lags[0]}"]
        logging.debug("Lag features tail:\n%s", df[cols].tail(3))
    return df


def add_calendar_features(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Encode day-of-week & month cycles as sine/cosine."""
    df = df.copy()
    df["dow"] = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    if debug:
        logging.debug("Calendar features head:\n%s", df[["dow", "dow_sin", "month_sin"]].head(3))
    return df


def save_diagnostics_pdf(
    series: pd.Series,
    pdf_path: Path,
    period: int = 5,
    lags: int = 30,
    model: str = "additive",
) -> pd.DataFrame:
    """
    Write a PDF with decomposition, ACF/PACF of residuals, and stationarity tests.
    Returns the decomposition DataFrame.
    """
    decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
    residuals = decomposition.resid.dropna()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # decomposition plot
        fig = decomposition.plot()
        fig.suptitle(f"{series.name} - {model.capitalize()} Decomposition", y=1.02)
        pdf.savefig(fig); plt.close(fig)

        # ACF/PACF of residuals
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        plot_acf(residuals, lags=lags, ax=axes[0]); axes[0].set_title("Residuals ACF")
        plot_pacf(residuals, lags=lags, ax=axes[1]); axes[1].set_title("Residuals PACF")
        pdf.savefig(fig); plt.close(fig)

        # Rolling stats + stationarity
        for data, title in [(series.dropna(), series.name), (residuals, "Residuals")]:
            rm = data.rolling(30, min_periods=1).mean()
            rs = data.rolling(30, min_periods=1).std()
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(data, label="Series"); ax.plot(rm, label="Rolling Mean"); ax.plot(rs, label="Rolling Std")
            ax.set_title(f"{title} Rolling Stats"); ax.legend()
            pdf.savefig(fig); plt.close(fig)

            adf_p = adfuller(data)[1]
            kpss_p = kpss(data, regression="c", nlags="auto")[1]
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.axis("off")
            txt = (
                f"{title} ADF p: {adf_p:.4f} ({'Stat.' if adf_p<0.05 else 'Non-Stat.'})\n"
                f"{title} KPSS p: {kpss_p:.4f} ({'Non-Stat.' if kpss_p<0.05 else 'Stat.'})"
            )
            ax.text(0.1, 0.5, txt, fontsize=10)
            pdf.savefig(fig); plt.close(fig)

    return pd.DataFrame({
        "seasonal": decomposition.seasonal,
        "trend": decomposition.trend,
        "residuals": decomposition.resid,
    })


def prepare_merged_data(
    price_path: Path,
    sentiment_path: Path,
    output_path: Path,
    halflife: int,
    lags: list[int],
    debug: bool = False,
) -> pd.DataFrame:
    """
    Build and save merged data:
      1. load price & sentiment
      2. merge
      3. impute sentiment via EWMA
      4. add price, lag, calendar, sentiment features
      5. drop rows with any NaN in crucial columns
      6. save to output_path
    """
    price = load_price_data(price_path)
    sentiment = load_sentiment_data(sentiment_path)
    df = merge_data(price, sentiment, debug)

    # sentiment imputation
    df["sentiment"], df["sent_missing"] = time_decay_impute(df["sentiment"], halflife=halflife)
    if debug:
        logging.debug("After sentiment impute, head:\n%s", df[["sentiment","sent_missing"]].head(3))

    # feature expansions
    df = add_price_features(df, debug=debug)
    df = add_lag_features(df, lags=lags, debug=debug)
    df = add_calendar_features(df, debug=debug)
    # rolling sentiment statistics
    df["sent_ma3"] = df["sentiment"].rolling(3, min_periods=1).mean()
    df["sent_ma7"] = df["sentiment"].rolling(7, min_periods=1).mean()
    df["sent_x_vol"] = df["sentiment"] * df["volatility"]

    # drop any rows with missing target or sentiment
    required = ["Close", "log_return", "sentiment"]
    df = df.dropna(subset=required).sort_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logging.info("Saved merged data with %d rows to %s", len(df), output_path)
    return df


def main():
    p = argparse.ArgumentParser(description="Prepare NVDA merged features for modeling")
    p.add_argument("--price-csv",       type=Path, default=Path("data") / "nvda_stock.csv")
    p.add_argument("--sentiment-csv",   type=Path, default=Path("data") / "nvda_sentiment_daily.csv")
    p.add_argument("--output-csv",      type=Path, default=Path("data") / "nvda_merged.csv")
    p.add_argument("--halflife",        type=int,   default=3, help="EWMA halflife for sentiment impute")
    # p.add_argument("--lags",            nargs="+",   type=int, default=[1,3,5,10,252])
    p.add_argument("--lags", nargs="+", type=int,
               default=[1, 2, 3, 5, 10, 252])   # ← added 2

    p.add_argument("--diagnostics-pdf", type=Path, default=Path("models") / "diagnostics.pdf")
    p.add_argument("--debug",           action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    df = prepare_merged_data(
        price_path=args.price_csv,
        sentiment_path=args.sentiment_csv,
        output_path=args.output_csv,
        halflife=args.halflife,
        lags=args.lags,
        debug=args.debug,
    )

    # optional diagnostics
    df_decomp = save_diagnostics_pdf(
        series=df["log_return"],
        pdf_path=args.diagnostics_pdf,
        period=5,
        lags=30,
        model="additive",
    )
    # If you want the decomposition as features, you can join here:
    # df.join(df_decomp).to_csv(args.output_csv, index=True)


if __name__ == "__main__":
    main()
