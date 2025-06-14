import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def analyze_sentiment(
    input_csv="data/nvda_news_google.csv",
    output_csv="data/nvda_sentiment_daily.csv"
):
    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return None

    df = pd.read_csv(input_csv)

    if "title" not in df.columns or "date" not in df.columns:
        print("❌ CSV must contain 'title' and 'date' columns.")
        return None

    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["date"] = pd.to_datetime(df["date"])

    daily_sentiment = df.groupby("date").agg({"sentiment": "mean"}).reset_index()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    daily_sentiment.to_csv(output_csv, index=False)

    print(f"✅ Saved daily sentiment to {output_csv}")
    return daily_sentiment

if __name__ == "__main__":
    analyze_sentiment()
