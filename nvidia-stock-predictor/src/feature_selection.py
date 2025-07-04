import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: Path) -> pd.DataFrame:
    """Load preprocessed NVDA data from CSV into a DataFrame indexed by Date."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    return df


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    annot: bool = False,
    cmap: str = 'coolwarm',
) -> None:
    """Plot and save a heatmap of the given matrix."""
    plt.figure(figsize=(12, 10))
    im = plt.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(matrix.columns))
    plt.xticks(ticks, matrix.columns, rotation=90)
    plt.yticks(ticks, matrix.index)
    if annot:
        # annotate each cell with its value
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{matrix.iat[i, j]:.2f}", ha='center', va='center', fontsize=6)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved heatmap to {output_path}")


def run_feature_selection(
    data_path: Path,
    out_dir: Path,
    exclude: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Perform correlation-based feature selection and save heatmaps."""
    df = load_data(data_path)
    target = 'log_return'

    # Identify feature columns dynamically
    features = [c for c in df.columns if c not in exclude]
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in data")
    if not features:
        raise ValueError("No feature columns left after exclusion")
    logging.info(f"Evaluating {len(features)} candidate features")

    # Correlation matrix for features
    feat_corr = df[features].corr()
    plot_heatmap(
        feat_corr,
        'Feature Correlation Heatmap',
        out_dir / 'feature_correlation_heatmap.png'
    )

    # Correlation with target
    corr_with_target = df[features + [target]].corr()[target].drop(target)
    corr_abs = corr_with_target.abs().sort_values(ascending=False)
    logging.info("Top features by |Pearson corr| with target:")
    for feat, val in corr_abs.head(top_k).items():
        logging.info(f"  {feat}: {val:.4f}")

    # Select top_k features
    top_feats = corr_abs.head(top_k).index.tolist()
    sub_corr = feat_corr.loc[top_feats, top_feats].abs()

    # Compute redundancy-adjusted scores
    max_inter = sub_corr.apply(lambda row: row.drop(row.name).max(), axis=1)
    scores = corr_abs[top_feats] - max_inter
    selection = scores.sort_values(ascending=False)

    logging.info("Top features after redundancy adjustment:")
    for feat, score in selection.head(int(top_k/2)).items():
        logging.info(
            f"  {feat}: corr={corr_with_target[feat]:.4f}, "
            f"max_inter={max_inter[feat]:.4f}, score={score:.4f}"
        )

    # Save heatmap of selected subset
    plot_heatmap(
        sub_corr,
        f'Top-{top_k} Features Inter-Correlation',
        out_dir / 'top_features_inter_corr_heatmap.png',
        annot=True
    )

    # Return DataFrame of selection scores
    result_df = pd.DataFrame({
        'feature': selection.index,
        'corr_with_target': corr_with_target[selection.index],
        'max_inter_corr': max_inter[selection.index],
        'score': selection.values,
    })
    result_df.to_csv(out_dir / 'feature_selection_scores.csv', index=False)
    logging.info(f"Saved feature selection scores to {out_dir / 'feature_selection_scores.csv'}")
    return result_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature selection for NVDA log returns"
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'data' / 'nvda_merged.csv',
        help='Path to merged data CSV'
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'models',
        help='Directory to save plots and results'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=[
            'log_return', 'Close', 'close_diff', 'simple_return', 'rsi14',
            'Volume', 'volatility', 'vol_roll5', 'close_l1', 'close_l3',
            'close_l5', 'close_l10', 'close_l252'
        ],
        help='Columns to exclude from candidate features'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top features by abs(correlation) to consider'
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_feature_selection(
        data_path=args.data,
        out_dir=args.out_dir,
        exclude=args.exclude,
        top_k=args.top_k,
    )
