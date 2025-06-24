import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Adjust SCRIPT_DIR
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load preprocessed NVDA data."""
    if path is None:
        path = os.path.join(SCRIPT_DIR, '..', 'data', 'nvda_merged.csv')
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    return df


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: str) -> None:
    """Plot and save a heatmap."""
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


def run_feature_selection() -> None:
    """Perform correlation analysis for feature selection."""
    df = load_data()
    target = 'log_return'

    # Identify feature columns (exclude target and price columns)
    exclude_cols = [
        target, 'Close', 'close_diff', 'simple_return', 'rsi14', 'Volume',
        'volatility', 'vol_roll5', 'close_l1', 'close_l10', 'close_l252',
        'close_l5', 'close_l3'
    ]
    features = [col for col in df.columns if col not in exclude_cols]

    # 1) Feature-Feature correlation
    feat_corr = df[features].corr()
    plot_heatmap(
        feat_corr,
        'Feature Correlation Heatmap',
        os.path.join(SCRIPT_DIR, '..', 'models', 'feature_correlation_heatmap.png')
    )
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
    plot_heatmap(
        sub_corr,
        'Top Features Inter-Correlation Heatmap',
        os.path.join(SCRIPT_DIR, '..', 'models', 'top_features_inter_corr_heatmap.png')
    )
    print('Top features inter-correlation heatmap saved.')


if __name__ == '__main__':
    run_feature_selection()