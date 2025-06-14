 NVIDIA Stock Price Prediction – Project Summary
We built a stock prediction system for NVIDIA that combines traditional time-series modeling, sentiment analysis from financial news, and deep learning.

Steps:

News Sentiment Analysis

Scraped NVIDIA-related news (2022–2025) from Yahoo Finance.

Applied VADER sentiment scoring to each headline.

Aggregated daily sentiment scores.

Data Merging & Feature Engineering

Merged stock price data with sentiment scores.

Created lag features for both price and sentiment.

Model Training & Evaluation
We trained three models to predict closing prices:

Linear Regression (used lag features + sentiment):

MAE: 1.87, MSE: 5.70 ✅ Best performance

ARIMA (time-series only):

MAE: 9.13, MSE: 118.81

LSTM (deep learning):

MAE: 10.28, MSE: 159.52

Key Insight:
Simple models with engineered features (like Linear Regression) outperformed complex ones (like LSTM) on our dataset. This shows that combining sentiment with structured price data can be very effective — even without deep models.
