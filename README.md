# Stock Market Prediction

This repository contains the code used for predicting NVIDIA's stock price using a mix of traditional time series models, sentiment extracted from financial news and a deep learning approach. The main project lives inside the `nvidia-stock-predictor` directory.

## Repository layout

```
.
├── LICENSE
├── nvidia-stock-predictor
│   ├── README.md                # short project summary
│   ├── data/                    # CSV data files and model outputs
│   ├── models/                  # generated plots
│   ├── requirements.txt         # dependencies
│   └── src/                     # all source code
└── README.md                    # (this file)
```

### Data folder

The `data/` directory already contains example CSVs such as `nvda_stock.csv` with historical prices, `nvda_sentiment_daily.csv` with aggregated daily sentiment scores, and prediction outputs produced by the various models.

### Source code

All scripts reside in `nvidia-stock-predictor/src/`. Below is an overview:

- `fetch_data.py` – download NVDA prices and news headlines.
- `scrape_yahoo_news.py` – scrape additional news articles via RSS.
- `sentiment_analysis.py` – run VADER sentiment on headlines and aggregate scores.
- `feature_engineering_reworked.py` – build the final merged feature set (prices, sentiment, technical indicators, calendar effects).
- `feature_selection.py` – simple correlation based feature analysis.
- `linear_regression_final.py` – final linear regression benchmark.
- `s_arima_final.py` – final SARIMAX benchmark.
- `lstm_final.py` – LSTM benchmark implemented in PyTorch.
- `obsolete/` – older prototype scripts kept for reference.

The remainder of this README dives deeper into the most important scripts.

## Key modules

### `feature_engineering_reworked.py`
Utility functions to create the modelling dataset:

1. **Load price and sentiment data** – `load_price_data()` and `load_sentiment_data()` read the raw CSVs and ensure consistent date indices.
2. **Merge datasets** – `merge_data()` joins prices with sentiment on the date index.
3. **Impute missing sentiment** – `time_decay_impute()` fills gaps using an exponentially weighted average so recent news has more impact.
4. **Add price‑based features** – `add_price_features()` computes returns, volatility, RSI and Bollinger bands.
5. **Create lags** – `add_lag_features()` generates lagged versions of returns, sentiment and price levels for the models.
6. **Calendar encoding** – `add_calendar_features()` adds day‑of‑week and month cycles via sine/cosine transforms.
7. **Diagnostics** – helper functions such as `check_stationarity()` and `save_diagnostics_pdf()` produce plots for stationarity and seasonality checks.

Running the script builds the full merged dataset and stores it in `data/nvda_merged.csv`.

### `feature_selection.py`
Performs a simple correlation analysis on the engineered dataset. It loads `nvda_merged.csv`, computes feature–feature and feature–target correlations and saves heatmaps to the `models/` folder. The printed rankings were used to decide which features to feed into the models.

### `linear_regression_final.py`
Implements the final Ordinary Least Squares benchmark. It mirrors the information used by the ARIMAX model and additionally adds technical indicators (RSI and several moving averages). The script discovers the AR lag order via `pmdarima.auto_arima`, trains on 80% of the data and evaluates on the remainder. At the end of the file are the recorded performances.

**Without sentiment**
```
Price MAE   : 1.8237
Price MSE   : 7.1993
Price RMSE  : 2.6832
Price MAPE  : 1.50%
Price SMAPE : 1.49%
Price R²    : 0.9510
Gap  MSE    : 11.368401
Gap  RMSE   : 3.371706
Baseline Gap MSE (zero): 11.312550
Directional Accuracy: 50.000%
```

**With sentiment**
```
Price MAE   : 1.7329
Price MSE   : 6.3465
Price RMSE  : 2.5192
Price MAPE  : 1.42%
Price SMAPE : 1.42%
Price R²    : 0.9572
Gap  MSE    : 10.741302
Gap  RMSE   : 3.277393
Baseline Gap MSE (zero): 10.664997
Directional Accuracy: 48.462%
```

### `s_arima_final.py`
Trains a SARIMAX model (non‑seasonal ARIMA with exogenous regressors). The script auto‑tunes the ARIMA order on the training set and optionally includes daily sentiment and technical indicators as exogenous variables. Results printed at the bottom of the file show the impact of sentiment.

**Without sentiment**
```
Price MAE   : 1.8053
Price MSE   : 7.0593
Price RMSE  : 2.6569
Price MAPE  : 1.48%
Price SMAPE : 1.48%
Price R²    : 0.9518
Gap  MSE    : 10.942012
Gap  RMSE   : 3.307871
Baseline Gap MSE (zero): 10.938747
```

**With sentiment**
```
Price MAE   : 1.7073
Price MSE   : 6.1743
Price RMSE  : 2.4848
Price MAPE  : 1.40%
Price SMAPE : 1.40%
Price R²    : 0.9580
Gap  MSE    : 10.223846
Gap  RMSE   : 3.197475
Baseline Gap MSE (zero): 10.237654
```

A full SARIMAX summary table (coefficients and diagnostics) is included in the file comments for reference.

### `lstm_final.py`
Contains a small PyTorch LSTM that operates on rolling windows of features. Sequences are built from lagged returns, technical indicators and optionally sentiment. Training uses early stopping on validation loss. The script saves predictions to CSV and plots predicted vs. actual prices. The bottom of the file records the following performance for the run **without** sentiment:

```
Price MAE   : 2.7116
Price MSE   : 13.0437
Price RMSE  : 3.6116
Price MAPE  : 2.20%
Price SMAPE : 2.19%
Price R²    : 0.9116
Gap  MSE    : 11.347156
Gap  RMSE   : 3.368554
Baseline Gap MSE (zero): 11.343951
Directional Accuracy: 47.407%
```

```
LSTM benchmark (with sentiment)
Price MAE:                  2.7000
Price MSE:                  12.2488
Price RMSE:                 3.4998
Price MAPE:                 2.19%
Price SMAPE:                2.18%
Price R²:                   0.9178
Gap  MSE:                   10.693910
Gap  RMSE:                  3.270154
Baseline Gap MSE (zero):    10.692839
Directional Accuracy:       48.837%
```

## Running the project

Typical workflow:

1. Run `fetch_data.py` and `scrape_yahoo_news.py` to gather raw data.
2. Execute `sentiment_analysis.py` to compute daily sentiment scores.
3. Use `feature_engineering_reworked.py` to create `nvda_merged.csv`.
4. Optionally run `feature_selection.py` for exploratory correlation analysis.
5. Train models via `linear_regression_final.py`, `s_arima_final.py` and `lstm_final.py`.

The produced plots and prediction CSVs will appear in the `models/` and `data/` directories respectively.