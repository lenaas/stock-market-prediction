import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access the key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_nvda_stock(start="2022-01-01", end="2024-12-31"):
    ticker = "NVDA"
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df.to_csv("data/nvda_stock.csv", index=False)
    print(f"Saved stock data to data/nvda_stock.csv")
    return df

def fetch_nvda_news(query="NVIDIA", from_date="2022-01-01", to_date="2024-12-31"):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    all_articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language='en',
        sort_by='relevancy',
        page_size=100  # adjust if needed
    )
    articles = all_articles["articles"]
    df = pd.DataFrame([{
        "date": a["publishedAt"][:10],
        "title": a["title"],
        "source": a["source"]["name"]
    } for a in articles])
    df.to_csv("data/nvda_news.csv", index=False)
    print(f"Saved news data to data/nvda_news.csv")
    return df

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    fetch_nvda_stock()
    fetch_nvda_news(NEWS_API_KEY)
