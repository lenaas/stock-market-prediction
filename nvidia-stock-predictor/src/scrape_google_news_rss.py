#!/usr/bin/env python3
"""
scrape_google_news_rss.py

Fetch headlines from Google News RSS for a given query & date range,
deduplicate, and save to CSV.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import feedparser
import pandas as pd


def scrape_google_news_rss(
    query: str,
    start: str,
    end: str,
    output_path: Path
) -> pd.DataFrame:
    """
    Scrape Google News RSS for `query` between `start` and `end` dates.
    
    Returns a DataFrame with columns [date, title, url], deduped.
    """
    # Build the RSS URL
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={query}+after:{start}+before:{end}"
        "&hl=en-US&gl=US&ceid=US:en"
    )
    logging.info("Fetching RSS feed from %s", rss_url)
    feed = feedparser.parse(rss_url)
    if feed.bozo:
        logging.warning("Encountered a parse error: %s", feed.bozo_exception)

    records = []
    for entry in feed.entries:
        try:
            pub = datetime(*entry.published_parsed[:6]).date()
        except Exception:
            # Skip entries without valid published date
            continue
        # Only include entries within the requested window
        if not (datetime.fromisoformat(start).date() <= pub <= datetime.fromisoformat(end).date()):
            continue
        records.append({
            "date": pub.isoformat(),
            "title": entry.title or "",
            "url":   entry.link    or ""
        })

    if not records:
        logging.warning("No articles found for query='%s' between %s and %s", query, start, end)

    df = pd.DataFrame(records)
    # drop exact duplicates by URL first, then by title
    before = len(df)
    df = df.drop_duplicates(subset="url").drop_duplicates(subset="title").reset_index(drop=True)
    logging.info("Deduplicated: %d → %d articles", before, len(df))

    # Ensure output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Saved %d articles to %s", len(df), output_path)
    return df


def main():
    p = argparse.ArgumentParser(
        description="Scrape Google News RSS and save headlines to CSV"
    )
    p.add_argument("--query",       type=str,   default="NVIDIA",
                   help="Search query for Google News")
    p.add_argument("--start",       type=str,   default="2022-01-01",
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",         type=str,   default="2025-12-31",
                   help="End date (YYYY-MM-DD)")
    p.add_argument("--output-csv",  type=Path, default=Path("data") / "nvda_news_google.csv",
                   help="Path to save the CSV")
    p.add_argument("--verbose",     action="store_true",
                   help="Enable debug-level logging")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    scrape_google_news_rss(
        query=args.query,
        start=args.start,
        end=args.end,
        output_path=args.output_csv
    )


if __name__ == "__main__":
    main()
