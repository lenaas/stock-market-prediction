"""Train an LSTM model on NVDA price and news sentiment data."""

import os
import random

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_sequences(features: np.ndarray, target: np.ndarray, seq_length: int):
    """Create sliding window sequences for LSTM input."""

    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm_model(seq_len: int = 10) -> None:
    """Train the LSTM model and save the results."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "nvda_merged.csv")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx - seq_len :]

    close_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    close_scaler.fit(train_df[["Close"]])
    sentiment_scaler.fit(train_df[["sentiment"]])
    target_scaler.fit(train_df[["log_return"]])

    def scale_features(df_part: pd.DataFrame) -> np.ndarray:
        close_scaled = close_scaler.transform(df_part[["Close"]])
        sentiment_scaled = sentiment_scaler.transform(df_part[["sentiment"]])
        return np.hstack([close_scaled, sentiment_scaled])

    train_features = scale_features(train_df)
    test_features = scale_features(test_df)

    train_target = target_scaler.transform(train_df[["log_return"]])
    test_target = target_scaler.transform(test_df[["log_return"]])

    X_train, y_train = create_sequences(train_features, train_target, seq_len)
    X_test, y_test = create_sequences(test_features, test_target, seq_len)

    val_size = max(1, int(len(X_train) * 0.1))
    train_X, val_X = X_train[:-val_size], X_train[-val_size:]
    train_y, val_y = y_train[:-val_size], y_train[-val_size:]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        ),
        batch_size=16,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(val_X, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        ),
        batch_size=16,
        shuffle=False,
    )

    class LSTMModel(torch.nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = torch.nn.Dropout(0.2)
            self.fc = torch.nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            return self.fc(out)

    model = LSTMModel(input_size=2)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float("inf")
    patience = 10
    epochs_without_improve = 0
    best_state = None

    for _ in range(50):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                val_losses.append(criterion(output, y_batch).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improve = 0
            best_state = model.state_dict()
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f"\n🧠 LSTM Results — MAE: {mae:.6f}, MSE: {mse:.6f}")

    model_path = os.path.join(model_dir, "lstm_model.pth")
    pred_path = os.path.join(model_dir, "lstm_predictions.csv")
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(
        {"actual": y_test_inv.flatten(), "predicted": y_pred_inv.flatten()}
    ).to_csv(pred_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted (LSTM)")
    plt.title("LSTM: Actual vs Predicted log returns")
    plt.xlabel("Time")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "lstm_plot.png"))


if __name__ == "__main__":
    train_lstm_model()
