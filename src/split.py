import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_prices(path=None):
    """
    Load dataset/prices.csv reliably from anywhere.
    This file (split.py) lives in: project_root/src/
    dataset folder lives in: project_root/dataset/
    """
    # Folder where split.py is located
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # If no custom path is passed, construct:
    # project_root/dataset/prices.csv
    if path is None:
        path = os.path.normpath(os.path.join(this_dir, "..", "dataset", "prices.csv"))

    if not os.path.exists(path):
        print("Error: file not found:", path)
        return None

    return pd.read_csv(path)



# -------------------------
# Extract YHOO close prices
# -------------------------
def get_yhoo_close(df):
    YHOO_Data = df[df["symbol"] == "YHOO"]
    close_stocks = np.array(YHOO_Data["close"])
    close_stocks = close_stocks.reshape(len(close_stocks), 1)
    return close_stocks


# -------------------------
# Scaling (0 to 1)
# -------------------------
def scale_close_stocks(close_stocks):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_stocks)
    return scaler, scaled


# -------------------------
# Split 80 / 20
# -------------------------
def split_close_stocks(close_stocks):
    train_len = int(len(close_stocks) * 0.80)
    
    train = close_stocks[:train_len]
    print("Train shape:", train.shape)
    print(train)

    test = close_stocks[train_len:]
    print("\nTest shape:", test.shape)
    print(test)

    return train, test

def process(data, n_features):
    xdata, ydata = [], []
    for i in range(n_features, len(data)):
        xdata.append(data[i - n_features:i, 0])
        ydata.append(data[i, 0])
    return np.array(xdata), np.array(ydata)

def train_test_split(train, test, n_features):
    x_train, y_train = process(train, n_features)
    x_test, y_test = process(test, n_features)
    return x_train, y_train, x_test, y_test


# -------------------------
# MAIN (runs independently)
# -------------------------
def main():
    df = load_prices()
    if df is None:
        return

    close_stocks = get_yhoo_close(df)

    scaler, scaled_close = scale_close_stocks(close_stocks)

    train, test = split_close_stocks(scaled_close)

       # =============================
    # YOUR REQUIRED ADDED SECTION
    # =============================
    print("\nPreparing training and testing sequences using the last 4 days as input features...\n")
    x_train, y_train, x_test, y_test = train_test_split(train, test, 4)

    print("Input features for training (x_train):", x_train.shape)
    print("Target values for training (y_train):", y_train.shape)
    print("Input features for testing (x_test):", x_test.shape)
    print("Target values for testing (y_test):", y_test.shape)
    
    print("\nx_train DataFrame:")
    print(pd.DataFrame(x_train, columns=['x1', 'x2', 'x3', 'x4']))

    print("\ny_train DataFrame:")
    print(pd.DataFrame(y_train, columns=['y']))
    
    print("\nTrain-test sequence creation complete.")
    print("Next step: Run 'models/model_linear.py' to train and evaluate the prediction model.\n")


if __name__ == "__main__":
    main()
