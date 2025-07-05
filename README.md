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

### `feature_engineering.py`
The script works with log returns instead of raw closing prices. `check_stationarity()` applies ADF and KPSS tests and `save_diagnostics_pdf()` plots seasonal decomposition plus ACF/PACF to ensure the series is usable for ARIMA style models.

Financial indicators include simple and log returns, rolling volatility with a high-volatility flag, short term return acceleration, RSI(14) and Bollinger band width. Lagged versions of returns, sentiment and closing price capture autoregressive structure.

Day-of-week and month cycles are encoded via sine and cosine so the models can learn calendar effects.

### `feature_selection.py`
Performs a simple correlation analysis on the engineered dataset. It loads `nvda_merged.csv`, computes feature–feature and feature–target correlations and saves heatmaps to the `models/` folder. The printed rankings were used to decide which features to feed into the models.

### `linear_regression_final.py`
Implements the final Ordinary Least Squares benchmark. It predicts the one-day-ahead price gap (Close_{t+1} − Open_{t+1}) and reconstructs the close from that gap. The feature set mirrors the ARIMAX information plus technical indicators. The script discovers the AR lag order via `pmdarima.auto_arima`, trains on 80% of the data and evaluates on the remainder.

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
Trains an ARIMAX model using statsmodels' SARIMAX class. The target is again the next-day gap (Close_{t+1} − Open_{t+1}). `pmdarima.auto_arima` searches for the best (p,d,q) order with `seasonal=False`. ARIMAX is used as it conveniently handles exogenous regressors such as sentiment and technical indicators. Also used SARIMAX with `seasonal=True` but there were no changes in performance.

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
Contains a small PyTorch LSTM that operates on rolling windows of features. The network also forecasts the next-day gap (Close_{t+1} − Open_{t+1}) so that the closing price can be reconstructed. Sequences are built from lagged returns, technical indicators and optionally sentiment. Training uses early stopping on validation loss. The script saves predictions to CSV and plots predicted vs. actual prices.

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
## Informations about Evaluations

### Price MAE
**Formula**  
$$
\mathrm{MAE} \;=\; \frac{1}{n}\sum_{i=1}^{n}\bigl|y_i - \hat y_i\bigr|
$$

**Interpretation**  
Mean Absolute Error measures the average absolute deviation between your predicted prices \(\hat y_i\) and the true prices \(y_i\). Lower MAE indicates closer predictions on average and is less sensitive to large outliers than MSE.

---

### Price MSE
**Formula**  
$$
\mathrm{MSE} \;=\; \frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - \hat y_i\bigr)^{2}
$$

**Interpretation**  
Mean Squared Error penalizes larger errors more heavily (due to squaring). It represents the average of squared prediction errors, so it is sensitive to outliers.

---

### Price RMSE
**Formula**  
$$
\mathrm{RMSE} \;=\; \sqrt{\frac{1}{n}\sum_{i=1}^{n}\bigl(y_i - \hat y_i\bigr)^{2}}
\;=\;\sqrt{\mathrm{MSE}}
$$

**Interpretation**  
Root Mean Squared Error brings the error metric back to the original units (price). It is dominated by larger errors and gives a sense of the typical size of your prediction errors.

---

### Price MAPE
**Formula**  
$$
\mathrm{MAPE} \;=\; \frac{100\%}{n}\sum_{i=1}^{n}
\left|\frac{y_i - \hat y_i}{y_i}\right|
$$

**Interpretation**  
Mean Absolute Percentage Error expresses the average error as a percentage of the true price. It is scale-invariant but can become unstable or infinite when any \(y_i\) is near zero.

---

### Price SMAPE
**Formula**  
$$
\mathrm{SMAPE} \;=\; \frac{100\%}{n}\sum_{i=1}^{n}
\frac{\bigl|y_i - \hat y_i\bigr|}{\bigl(|y_i| + |\hat y_i|\bigr)/2}
$$

**Interpretation**  
Symmetric MAPE bounds the error between 0% and 200% by dividing by the average of actual and predicted values, mitigating extreme percentages when values approach zero.

---

### Price \(R^2\) (Coefficient of Determination)
**Formula**  
$$
R^2 \;=\; 1 \;-\;
\frac{\displaystyle\sum_{i=1}^{n}(y_i - \hat y_i)^{2}}
{\displaystyle\sum_{i=1}^{n}(y_i - \bar y)^{2}}
\quad\text{where}\quad
\bar y = \frac{1}{n}\sum_{i=1}^{n}y_i
$$

**Interpretation**  
Proportion of variance in the true prices explained by the model.  
- \(R^2 = 1\) indicates a perfect fit.  
- \(R^2 = 0\) means the model is no better than predicting the mean.  
- \(R^2 < 0\) means the model performs worse than the mean predictor.

---

### Gap MSE
**Definition of Gap**  
Let  
$$
g_i = y_i - y_{i-1}, 
\quad
\hat g_i = \hat y_i - y_{i-1}.
$$

**Formula**  
$$
\mathrm{Gap\ MSE} \;=\;
\frac{1}{n-1}\sum_{i=2}^{n}(g_i - \hat g_i)^{2}
$$

**Interpretation**  
Quantifies how well the model predicts the actual one‐step price change (gap). Lower values indicate more accurate sizing of price moves.

---

### Gap RMSE
**Formula**  
$$
\mathrm{Gap\ RMSE} \;=\;
\sqrt{\frac{1}{n-1}\sum_{i=2}^{n}(g_i - \hat g_i)^{2}}
\;=\;\sqrt{\mathrm{Gap\ MSE}}
$$

**Interpretation**  
Gives the typical magnitude of price-change prediction errors in the original price‐difference units.

---

## Baseline Gap MSE (zero)
**Formula**  
$$
\mathrm{Baseline\ Gap\ MSE}
\;=\;
\frac{1}{n-1}\sum_{i=2}^{n}g_i^{2}
$$

**Interpretation**  
The MSE obtained by always predicting zero change (no movement). Serves as a simple benchmark—your model should achieve a lower Gap MSE to be considered useful.

---

### Directional Accuracy
**Formula**  
$$
\mathrm{DirAcc} \;=\;
\frac{1}{n-1}\sum_{i=2}^{n}
\mathbf{1}\bigl[\operatorname{sign}(\hat g_i) = \operatorname{sign}(g_i)\bigr]
$$

**Interpretation**  
Fraction of time‐steps where the model correctly predicts the direction (up/down) of the price move. A value above 50% indicates better-than-random directional forecasting.



## Running the project

Typical workflow:

1. Run `fetch_data.py` and `scrape_yahoo_news.py` to gather raw data.
2. Execute `sentiment_analysis.py` to compute daily sentiment scores.
3. Use `feature_engineering_reworked.py` to create `nvda_merged.csv`.
4. Optionally run `feature_selection.py` for exploratory correlation analysis.
5. Train models via `linear_regression_final.py`, `s_arima_final.py` and `lstm_final.py`.

The produced plots and prediction CSVs will appear in the `models/` and `data/` directories respectively.