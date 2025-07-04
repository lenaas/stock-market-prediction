#!/usr/bin/env python3
"""
sentiment_analysis.py

Compute daily sentiment scores from a CSV of headlines.
Supports FinBERT (via HuggingFace) or VADER fallback.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

def analyze_with_vader(df: pd.DataFrame) -> pd.Series:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    logging.info("Using VADER for sentiment scoring")
    scores = df["title"].fillna("").astype(str).map(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    return scores

def analyze_with_finbert(df: pd.DataFrame, batch_size: int = 16) -> pd.Series:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading FinBERT model on {device}")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()

    texts = df["title"].fillna("").astype(str).tolist()
    scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

        # FinBERT order: [positive, negative, neutral]
        batch_scores = probs[:, 0] - probs[:, 1]
        scores.extend(batch_scores.tolist())

        logging.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size+1}")

    return pd.Series(scores, index=df.index)

def fill_daily_index(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Reindex to every calendar day in [start,end], forward-fill missing sentiment."""
    idx = pd.date_range(start, end, freq="D")
    daily = daily.set_index("date").reindex(idx)
    daily.index.name = "date"
    daily["sentiment"].ffill(inplace=True)
    return daily.reset_index()

def main():
    p = argparse.ArgumentParser(description="Compute daily sentiment from headlines CSV")
    p.add_argument("--input-csv",  type=Path, default=Path("data") / "nvda_news_google.csv")
    p.add_argument("--output-csv", type=Path, default=Path("data") / "nvda_sentiment_daily.csv")
    p.add_argument("--model",      choices=["finbert","vader"], default="finbert",
                   help="Which sentiment model to use")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for Transformer inference")
    p.add_argument("--verbose",    action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    if not args.input_csv.exists():
        logging.error("Input file not found: %s", args.input_csv)
        return

    df = pd.read_csv(args.input_csv, parse_dates=["date"])
    if not {"date","title"}.issubset(df.columns):
        logging.error("Input CSV must contain 'date' and 'title' columns")
        return

    # normalize dates to midnight and sort
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df.sort_values("date", inplace=True)

    # compute sentiment per headline
    if args.model == "finbert":
        try:
            df["sentiment"] = analyze_with_finbert(df, args.batch_size)
        except Exception as e:
            logging.warning("FinBERT failed (%s); falling back to VADER", e)
            df["sentiment"] = analyze_with_vader(df)
    else:
        df["sentiment"] = analyze_with_vader(df)

    # average by day
    daily = (
        df.groupby("date")["sentiment"]
          .mean()
          .reset_index()
    )

    # fill any missing days between min/max date
    start, end = daily["date"].min(), daily["date"].max()
    daily_full = fill_daily_index(daily, start, end)

    # ensure output dir exists
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_full.to_csv(args.output_csv, index=False)
    logging.info("Saved daily sentiment (%d days) to %s", len(daily_full), args.output_csv)

if __name__ == "__main__":
    main()
