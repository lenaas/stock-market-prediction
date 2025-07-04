import os
from pathlib import Path
from dotenv import load_dotenv
import yfinance as yf
from newsapi import NewsApiClient
import pandas as pd
import logging

# ——— Setup ———
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if NEWS_API_KEY is None:
    raise RuntimeError("Missing NEWS_API_KEY")

logging.basicConfig(level=logging.INFO)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_nvda_stock(ticker="NVDA", start="2022-01-01", end="2024-12-31"):
    logging.info(f"Downloading {ticker} from {start} to {end}")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    df = df.reset_index()
    out = DATA_DIR / f"{ticker}_stock.csv"
    df.to_csv(out, index=False)
    logging.info(f"Wrote stock data to {out}")
    return df

def fetch_nvda_news(query="NVIDIA", from_date="2022-01-01", to_date="2024-12-31"):
    logging.info(f"Fetching news for '{query}' from {from_date} to {to_date}")
    client = NewsApiClient(api_key=NEWS_API_KEY)
    all_articles = []
    page = 1
    while True:
        res = client.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="relevancy",
            page_size=100,
            page=page
        )
        articles = res.get("articles", [])
        if not articles:
            break
        all_articles.extend(articles)
        if len(articles) < 100:
            break
        page += 1

    df = pd.DataFrame([{
        "publishedAt": a["publishedAt"],
        "title": a["title"],
        "source": a["source"]["name"]
    } for a in all_articles])

    out = DATA_DIR / f"{query.lower()}_news.csv"
    df.to_csv(out, index=False)
    logging.info(f"Wrote {len(df)} news items to {out}")
    return df

if __name__ == "__main__":
    stock_df = fetch_nvda_stock()
    news_df  = fetch_nvda_news()
