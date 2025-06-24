import os
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from typing import List

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Adjust SCRIPT_DIR
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load preprocessed NVDA data with sentiment, log_return, close_l1, and Close columns."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    return df

def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: str) -> None:
    """Generic function to plot and save a heatmap of the given matrix."""
    plt.figure(figsize=(12, 10))
    im = plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(matrix.columns))
    plt.xticks(ticks, matrix.columns, rotation=90)
    plt.yticks(ticks, matrix.index)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def train_linear_model(df: pd.DataFrame) -> LinearRegression:
    """Train simple OLS on log returns using only sentiment, reconstruct Close price, and evaluate."""
    # Use only sentiment as predictor
    features = ['sentiment', 'sentiment_l252']
    target = 'log_return'

    # Drop rows with missing sentiment or required columns
    df_model = df[features + [target, 'close_l1', 'Close']].dropna()

    # Train-test split (80/20)
    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    prev_close = test_df['close_l1'].values
    true_close = test_df['Close'].values

    # Fit OLS regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict log returns
    preds_log = model.predict(X_test)

    # Reconstruct Close price forecasts: Close_t = Close_{t-1} * exp(log_return_t)
    preds_close = prev_close * np.exp(preds_log)

    # Compute evaluation metrics on prices
    mae_price = mean_absolute_error(true_close, preds_close)
    mse_price = mean_squared_error(true_close, preds_close)
    r2_price = r2_score(true_close, preds_close)

    print("\nLinear Regression (sentiment only)")
    print(f"MAE (Close price): {mae_price:.4f}")
    print(f"MSE (Close price): {mse_price:.4f}")
    print(f"R^2 (Close price): {r2_price:.4f}")
    print(f"Coefficient (sentiment): {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")

    # Plot actual vs predicted Close prices
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, true_close, label='Actual Close')
    plt.plot(test_df.index, preds_close, label='Predicted Close')
    plt.title('Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(SCRIPT_DIR, '..', 'models', 'linear_sentiment_l252_only.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    return model


def main():
    df = load_data()
    target = 'log_return'

    # Identify feature columns (exclude target and price columns)
    exclude_cols = [target, 'Close', 'close_diff', 'simple_return', 'rsi14', 'Volume', 'volatility', 'vol_roll5', 'close_l1', 'close_l10', 'close_l252', 'close_l5', 'close_l3']
    features = [col for col in df.columns if col not in exclude_cols]

    # 1) Feature-Feature correlation
    feat_corr = df[features].corr()
    plot_heatmap(feat_corr,
                 'Feature Correlation Heatmap',
                 os.path.join(SCRIPT_DIR, '..', 'models', 'feature_correlation_heatmap.png'))
    print('Feature-Feature correlation heatmap saved.')

    # 2) Feature-Target correlation
    corr_with_target = df[features + [target]].corr()[target].drop(target)
    corr_series = corr_with_target.abs().sort_values(ascending=False)
    print('\nTop features by correlation with target (log_return):')
    print(corr_series.head(10))
    # Result: 
    # month_sin         0.099607
    # bb_width          0.077009
    # log_return_l10    0.076243
    # Open              0.067336
    # log_return_l3     0.062217
    # sentiment_l1      0.058493
    # High              0.053978
    # sentiment_l10     0.050022
    # Low               0.048104
    # dow               0.046932

    # 3) Select top features minimizing inter-correlation
    top_feats = corr_series.head(10).index.tolist()
    sub_corr = feat_corr.loc[top_feats, top_feats].abs()

    # Compute each feature's max inter-feature correlation
    max_inter = sub_corr.apply(lambda row: row.drop(row.name).max(), axis=1)
    scores = corr_series[top_feats] - max_inter
    selection = scores.sort_values(ascending=False)

    print('\nFeatures with high target correlation and low inter-feature correlation:')
    for feat, score in selection.head(5).items():
        print(f"{feat}: target_corr={corr_with_target[feat]:.4f}, max_inter_corr={max_inter[feat]:.4f}, score={score:.4f}")
    # Result:
    # dow: target_corr=-0.0469, max_inter_corr=0.0422, score=0.0047
    # log_return_l3: target_corr=-0.0622, max_inter_corr=0.1468, score=-0.0846
    # log_return_l10: target_corr=-0.0762, max_inter_corr=0.1699, score=-0.0936
    # bb_width: target_corr=0.0770, max_inter_corr=0.3038, score=-0.2268
    # month_sin: target_corr=0.0996, max_inter_corr=0.3838, score=-0.2842

    # 4) Heatmap of top features inter-correlation
    plot_heatmap(sub_corr,
                 'Top Features Inter-Correlation Heatmap',
                 os.path.join(SCRIPT_DIR, '..', 'models', 'top_features_inter_corr_heatmap.png'))
    print('Top features inter-correlation heatmap saved.')

    #train_linear_model(df)
    


if __name__ == '__main__':
    main()


# Results
# Sentiment Only: MAE 2.8735, MSE 12.9655, R^2 0.9103, Coefficient sentiment -0.016435
# Sentiment + Lag 1 Sentiment: MAE 2.7662, MSE 12.7183, R^2 0.9120, Coefficient (sentiment) 0.026246
# Sentiment + Lagged Sentiments: MAE 3.1343, MSE 16.0328, R^2 0.8890, Coefficient (sentiment) 0.012388 
# Sentiment + Lag 3 Sentiments: MAE (Close price): 2.9469 MSE (Close price): 13.3302 R^2 (Close price): 0.9077 Coefficient (sentiment): -0.028210
# Sentiment + Lag 10 Sentiments: MAE (Close price): 3.1207 MSE (Close price): 15.1730 R^2 (Close price): 0.8950 Coefficient (sentiment): -0.040413
# Sentiment + Lag 252 Sentiment: MAE (Close price): 2.8735 MSE (Close price): 12.9655 R^2 (Close price): 0.9103 Coefficient (sentiment): -0.016435
# Best performed: Lag 1 and Lag 252 (the later on maybe only due to seasonal trend effects!)
