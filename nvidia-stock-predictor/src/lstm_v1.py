# lstm_same_info_pytorch.py  ────────────────────────────────────────────────
import os, math, warnings, random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ──────────────────────────────
# 0. reproducibility helper
# ──────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────
# 1. paths & data loader
# ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(SCRIPT_DIR, "..", "data", "nvda_merged.csv")

    df = (
        pd.read_csv(path, parse_dates=["Date"])
          .set_index("Date")
          .sort_index()
          .asfreq("B")
    )
    df.index.freq = "B"
    df[["close_l1", "Close"]] = df[["close_l1", "Close"]].fillna(method="bfill")
    return df


# ──────────────────────────────
# 2. create supervised sequences
# ──────────────────────────────
def make_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "log_return",
    seq_len: int = 5,
) -> Tuple[np.ndarray, ...]:
    X, y, close_t, dates = [], [], [], []
    for i in range(seq_len - 1, len(df) - 1):
        X.append(df[feature_cols].iloc[i - seq_len + 1 : i + 1].values)
        y.append(df[target_col].iloc[i + 1])       # predict t+1
        close_t.append(df["Close"].iloc[i])        # price at t
        dates.append(df.index[i + 1])              # date of t+1
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(close_t, dtype=np.float32),
        pd.DatetimeIndex(dates),
    )


# ──────────────────────────────
# 3. tiny LSTM in PyTorch
# ──────────────────────────────
class PriceLSTM(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden, 16)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # last time-step
        # out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.out(out).squeeze(1)   # (batch,)


# ──────────────────────────────
# 4. training / evaluation
# ──────────────────────────────
def train_lstm_same_info(
    df: pd.DataFrame,
    seq_len: int = 15,
    epochs: int = 500,
    batch_size: int = 32,
    patience: int = 30,
    seed: int = 0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ────────── feature engineering (same info set as ARIMAX) ──────────
    df_feat = df.copy()
    df_feat["sentiment_l1"] = df_feat["sentiment"].shift(1)
    for k in (1, 5, 10):
        df_feat[f"log_return_l{k}"] = df_feat["log_return"].shift(k)

    feature_cols = [
        "sentiment", "sentiment_l1",
        "log_return", "log_return_l1",
        "log_return_l5", "log_return_l10",
    ]
    df_feat = df_feat.dropna(subset=feature_cols + ["log_return", "Close"])

    # ────────── scale features AND target (fit on training only) ──────────
    split_idx = int(len(df_feat) * 0.8)
    scaler_x = StandardScaler().fit(df_feat[feature_cols].iloc[:split_idx])
    scaler_y = StandardScaler().fit(df_feat[["log_return"]].iloc[:split_idx])

    df_feat[feature_cols] = scaler_x.transform(df_feat[feature_cols])
    df_feat["log_return_scaled"] = scaler_y.transform(df_feat[["log_return"]])

    # ────────── build sequences ──────────
    X, y_scaled, close_t, dates = make_sequences(
        df_feat,
        feature_cols,
        target_col="log_return_scaled",
        seq_len=seq_len,
    )

    # chronological 80/20 split
    split_adj = split_idx - (seq_len - 1)
    X_train, X_test = X[:split_adj], X[split_adj:]
    y_train, y_test = y_scaled[:split_adj], y_scaled[split_adj:]
    close_t_train, close_t_test = close_t[:split_adj], close_t[split_adj:]
    dates_test = dates[split_adj:]

    # ────────── PyTorch datasets & loaders ──────────
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train))
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test), torch.tensor(y_test))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)

    # ────────── model & optimiser ──────────
    model = PriceLSTM(n_features=len(feature_cols)).to(device)
    criterion = torch.nn.MSELoss()  # still on *scaled* space for back-prop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    best_val, patience_cnt, best_state = np.inf, 0, None

    # ────────── training loop ──────────
    for epoch in range(1, epochs + 1):
        model.train()

        if epoch == 1:                      # only once, on the very first batch
            xb0, yb0 = next(iter(train_loader))
            xb0, yb0 = xb0.to(device), yb0.to(device)

            # forward
            pred0 = model(xb0)
            print(f"[DEBUG] raw preds (first 5): {pred0[:5].detach().cpu().numpy()}")

            loss0 = criterion(pred0, yb0)
            loss0.backward()

            # gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            print(f"[DEBUG] total grad-norm    : {grad_norm:.4e}")

            # parameter update magnitude we *would* have
            step_sizes = []
            for p in model.parameters():
                if p.grad is not None:
                    step_sizes.append((optimizer.param_groups[0]['lr'] * p.grad).norm().item())
            print(f"[DEBUG] mean |Δθ| per layer : {np.mean(step_sizes):.4e}")

            optimizer.zero_grad()           # clean up so the real epoch still runs
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ─── validation on *un-scaled* returns for interpretable metric ───
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds_scaled = model(xb).cpu().numpy().reshape(-1, 1)
                true_scaled  = yb.cpu().numpy().reshape(-1, 1)

                preds_r = scaler_y.inverse_transform(preds_scaled).flatten()
                true_r  = scaler_y.inverse_transform(true_scaled).flatten()
                val_losses.append(np.mean((preds_r - true_r) ** 2))

        val_loss = float(np.mean(val_losses))

        # ─── early-stopping bookkeeping ───
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping @ epoch {epoch}")
                break

        # scheduler *after* epoch (so lr logged corresponds to next epoch)
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d} | val MSE(unscaled)={val_loss:.6f} | lr={lr_now:.2e}")

    model.load_state_dict(best_state)

    # ────────── Test predictions ──────────
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test).to(device)).cpu().numpy()

    # inverse-scale preds to raw log-returns
    preds_log = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    true_log  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # reconstruct price_{t+1}
    preds_close = close_t_test * np.exp(preds_log)
    true_close  = df_feat["Close"].reindex(dates_test).values
###################################################################################################


    ######################################################################################################
    # ────────── metrics ──────────
    mae_p  = mean_absolute_error(true_close, preds_close)
    mse_p  = mean_squared_error(true_close, preds_close)
    rmse_p = math.sqrt(mse_p)
    mape_p = np.mean(np.abs((true_close - preds_close) / true_close)) * 100
    smape_p= np.mean(2 * np.abs(true_close - preds_close) /
                     (np.abs(true_close) + np.abs(preds_close))) * 100
    r2_p   = r2_score(true_close, preds_close)

    mse_r   = mean_squared_error(true_log, preds_log)
    rmse_r  = math.sqrt(mse_r)
    baseline_r = mean_squared_error(true_log, np.zeros_like(true_log))
    dir_acc = (np.sign(preds_log) == np.sign(true_log)).mean()

    print("\nLSTM (PyTorch) – same info set")
    print(f"→ Price MAE:                  {mae_p:.4f}")
    print(f"→ Price MSE:                  {mse_p:.4f}")
    print(f"→ Price RMSE:                 {rmse_p:.4f}")
    print(f"→ Price MAPE:                 {mape_p:.2f}%")
    print(f"→ Price SMAPE:                {smape_p:.2f}%")
    print(f"→ Price R²:                   {r2_p:.4f}")
    print(f"→ Return MSE:                 {mse_r:.6f}")
    print(f"→ Return RMSE:                {rmse_r:.6f}")
    print(f"→ Baseline Return MSE (zero): {baseline_r:.6f}")
    print(f"→ Directional Accuracy:       {dir_acc:.3%}")

    # ────────── Plot ──────────
    plt.figure(figsize=(10, 5))
    plt.plot(dates_test, true_close, label="Actual Close")
    plt.plot(dates_test, preds_close, label="Predicted Close (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("LSTM – Actual vs Forecast Close")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model


# ──────────────────────────────
def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = load_data()
    train_lstm_same_info(df)


if __name__ == "__main__":
    main()
