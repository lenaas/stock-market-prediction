import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)


def train_lstm_model():
    df = pd.read_csv("data/nvda_merged.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    # Separate scalers for each feature
    close_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()

    close_scaled = close_scaler.fit_transform(df[["Close"]])
    sentiment_scaled = sentiment_scaler.fit_transform(df[["sentiment"]])

    # Combine scaled features
    data_scaled = np.hstack([close_scaled, sentiment_scaled])

    # Sequence preparation
    SEQ_LEN = 10
    X, y = create_sequences(data_scaled, SEQ_LEN)

    # Train/test split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Model
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(SEQ_LEN, 2)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Training
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, callbacks=[es], verbose=0)

    # Prediction
    y_pred = model.predict(X_test)

    # Inverse transform predictions using only 'Close' scaler
    y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = close_scaler.inverse_transform(y_pred).flatten()

    # Evaluation
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f"\n🧠 LSTM Results — MAE: {mae:.2f}, MSE: {mse:.2f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted (LSTM)")
    plt.title("LSTM: Actual vs Predicted NVDA Close Price")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/lstm_plot.png")
    plt.show()


if __name__ == "__main__":
    train_lstm_model()
