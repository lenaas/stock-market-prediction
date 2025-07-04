#!/usr/bin/env python3
"""
main.py — orchestrate the full NVDA headline-sentiment → price-forecast workflow
"""

from __future__ import annotations
import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent          # repo root
SRC_DIR   = ROOT_DIR / "src"
DATA_DIR  = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

for _d in (DATA_DIR, MODELS_DIR):
    _d.mkdir(exist_ok=True, parents=True)

# add src/ to the module search path so "import fetch_data" works
sys.path.insert(0, str(SRC_DIR))

def _run(script: str, *extra_args: str) -> None:
    """
    Call a script inside src/ as a subprocess.
    Prints the command so you can see what's happening.
    """
    cmd = [sys.executable, str(SRC_DIR / script), *extra_args]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ──────────────────────────────────────────────
# 1) RAW DATA
# ──────────────────────────────────────────────
def step_fetch_stock(ticker: str, start: str, end: str) -> None:
    print("\n📥  Fetching daily stock prices …")
    from fetch_data import fetch_nvda_stock        # resolved via sys.path
    fetch_nvda_stock(ticker=ticker, start=start, end=end)

def step_scrape_rss(query: str, start: str, end: str) -> Path:
    print("\n🗞️  Scraping Google-News RSS …")
    from scrape_google_news_rss import scrape_google_news_rss  # noqa: E402
    out = DATA_DIR / f"{query.lower().replace(' ', '_')}_news_google.csv"
    scrape_google_news_rss(query=query, start=start, end=end, output_path=out)
    return out

# ──────────────────────────────────────────────
# 2) SENTIMENT
# ──────────────────────────────────────────────
def step_sentiment(input_csv: Path, model: str, batch: int) -> None:
    print("\n💬  Computing daily sentiment …")
    _run(
        "sentiment_analysis.py",
        "--input-csv", str(input_csv),
        "--output-csv", str(DATA_DIR / "nvda_sentiment_daily.csv"),
        "--model", model,
        "--batch-size", str(batch),
    )

# ──────────────────────────────────────────────
# 3) FEATURES
# ──────────────────────────────────────────────
def step_feature_engineering() -> None:
    print("\n🔧  Engineering technical features …")
    _run(
        "feature_engineering.py",
        "--price-csv", str(DATA_DIR / "nvda_stock.csv"),
        "--sentiment-csv", str(DATA_DIR / "nvda_sentiment_daily.csv"),
        "--output-csv", str(DATA_DIR / "nvda_merged.csv"),
    )

def step_feature_selection(top_k: int) -> None:
    if top_k <= 0:
        print("\n🚸  Feature-selection skipped (top_k ≤ 0)")
        return
    print("\n🎯  Correlation-based feature pruning …")
    _run(
        "feature_selection.py",
        "--data", str(DATA_DIR / "nvda_merged.csv"),
        "--out-dir", str(MODELS_DIR),
        "--top-k", str(top_k),
    )

# ──────────────────────────────────────────────
# 4) MODELLING
# ──────────────────────────────────────────────
def step_train_models(skip_arima: bool) -> None:
    print("\n🤖  Training benchmark models …")
    _run("linear_regression_v1.py")
    _run("lstm_v1.py")

    if not skip_arima:
        _run("s_arima_v1.py")

    # user-supplied trainers (auto-detect)
    for extra in ("train_arima.py", "train_linear_arima.py", "train_lstm.py"):
        if (SRC_DIR / extra).exists():
            _run(extra)

# ──────────────────────────────────────────────
# 5) EVALUATION
# ──────────────────────────────────────────────
def step_evaluate() -> None:
    print("\n📊  Consolidating forecasts & metrics …")
    _run("evaluate_models.py")

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    today = _dt.date.today().isoformat()
    p = argparse.ArgumentParser(description="Run the full NVDA price-forecast pipeline")
    p.add_argument("--ticker", default="NVDA", help="Yahoo ticker symbol")
    p.add_argument("--query",  default="NVIDIA", help="News search string")
    p.add_argument("--start",  default="2022-01-01", help="YYYY-MM-DD")
    p.add_argument("--end",    default=today, help="YYYY-MM-DD")
    p.add_argument("--sentiment-model", choices=["finbert", "vader"], default="finbert")
    p.add_argument("--batch-size", type=int, default=16, metavar="N")
    p.add_argument("--top-k", type=int, default=20,
                   help="Keep the |corr| top-k features (≤0 to skip)")
    p.add_argument("--skip-arima", action="store_true",
                   help="Skip built-in ARIMAX / SARIMAX trainer")
    return p.parse_args()

# ──────────────────────────────────────────────
# entry
# ──────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    # 1) raw data
    step_fetch_stock(args.ticker, args.start, args.end)
    news_csv = step_scrape_rss(args.query, args.start, args.end)

    # 2) sentiment
    step_sentiment(news_csv, args.sentiment_model, args.batch_size)

    # 3) features
    step_feature_engineering()
    step_feature_selection(args.top_k)

    # 4) modelling
    step_train_models(args.skip_arima)

    # 5) evaluation
    step_evaluate()

    print("\n✅  Pipeline completed. Artefacts are in ./data and ./models")

if __name__ == "__main__":
    main()
