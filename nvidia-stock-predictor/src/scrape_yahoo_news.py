import feedparser
import pandas as pd
from datetime import datetime
import os

def scrape_google_news_rss(query="NVIDIA", output_path="data/nvda_news_google.csv"):
    rss_url = f"https://news.google.com/rss/search?q={query}+after:2022-01-01+before:2025-12-31&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    articles = []
    for entry in feed.entries:
        articles.append({
            "date": datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d"),
            "title": entry.title,
            "url": entry.link
        })

    if not os.path.exists("data"):
        os.makedirs("data")

    df = pd.DataFrame(articles)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} articles to {output_path}")

if __name__ == "__main__":
    scrape_google_news_rss()
