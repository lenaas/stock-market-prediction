import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# def analyze_sentiment(
#     input_csv="data/nvda_news_google.csv",
#     output_csv="data/nvda_sentiment_daily.csv"
# ):
#     if not os.path.exists(input_csv):
#         print(f"❌ Input file not found: {input_csv}")
#         return None

#     df = pd.read_csv(input_csv)

#     if "title" not in df.columns or "date" not in df.columns:
#         print("❌ CSV must contain 'title' and 'date' columns.")
#         return None

#     analyzer = SentimentIntensityAnalyzer()
#     df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
#     df["date"] = pd.to_datetime(df["date"])

#     daily_sentiment = df.groupby("date").agg({"sentiment": "mean"}).reset_index()
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     daily_sentiment.to_csv(output_csv, index=False)

#     print(f"✅ Saved daily sentiment to {output_csv}")
#     return daily_sentiment

# if __name__ == "__main__":
#     analyze_sentiment()



import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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

    # Load FinBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def get_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # Get probabilities for each class (positive, negative, neutral)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # You can customize how you want to calculate the final sentiment score.
        # For simplicity, we can use a weighted average of the probabilities.
        # The labels are typically ordered as [positive, negative, neutral]
        # A positive score is desirable.
        sentiment_score = probabilities[0][0] - probabilities[0][1] 
        return sentiment_score.item()

    df["sentiment"] = df["title"].apply(lambda x: get_sentiment(str(x)))
    df["date"] = pd.to_datetime(df["date"])

    daily_sentiment = df.groupby("date").agg({"sentiment": "mean"}).reset_index()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    daily_sentiment.to_csv(output_csv, index=False)

    print(f"✅ Saved daily sentiment to {output_csv}")
    return daily_sentiment

if __name__ == "__main__":
    analyze_sentiment()