import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Utility: load prices CSV
# -------------------------
def load_prices(path=os.path.join("..", "dataset", "prices.csv")):
    if not os.path.exists(path):
        print("Error: file not found:", path)
        return None
    df = pd.read_csv(path)
    return df


def check_company_counts(df):

    # --- your original count print ---
    counts = df[
        (df['symbol'] == 'ADBE') |
        (df['symbol'] == 'FB')   |
        (df['symbol'] == 'GS')   |
        (df['symbol'] == 'MSFT') |
        (df['symbol'] == 'XRX')  |
        (df['symbol'] == 'YHOO')
    ]['symbol'].value_counts()

    counts.index.name = None
    counts.name = "symbol"
    print(counts)
    print("Maximum number of rows:", max(df['symbol'].value_counts()))

    # --- your next print text ---
    print("\nSince ADBE,GS,MSFT,XRX,and YHOO have "
          "maximum data, we can build a model for any of them "
          "\nBuilding a model for YHOO company to predict closeing stock\n")

    # --- your YHOO selection ---
    YHOO_Data = df[df['symbol'] == 'YHOO']
    print(YHOO_Data)  # display() replaced with print()

    # --- your explanation ---
    print("\nTo presict the closeing stock we required closeing stock of past "
          "few days so we need only closeing stock column")

    # --- your close-stocks extraction ---
    close_stocks = np.array(YHOO_Data.loc[:, 'close'])
    print(close_stocks)

    close_stocks = close_stocks.reshape(len(close_stocks), 1)
    print(close_stocks.shape)
    print(close_stocks)

    # return useful items
    return YHOO_Data, close_stocks


def scale_close_stocks(close_stocks):
    """
    Scales close_stocks between 0 and 1 using MinMaxScaler.
    Returns scaler and scaled array.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_stocks_scaled = scaler.fit_transform(close_stocks)
    
    print("\nscaling features between 0 and 1")
    print(close_stocks_scaled)   # your original print statement
    
    return scaler, close_stocks_scaled


def main():
    df = load_prices()
    if df is None:
        return
    
    # Get YHOO data + close_stocks array
    YHOO_Data, close_stocks = check_company_counts(df)

    # Now apply scaling
    scaler, close_stocks_scaled = scale_close_stocks(close_stocks)

    print("\nData preparation and scaling are complete.")
    print("Next step: Run 'split.py' to create sequences and finalize train/test data.\n")

    
    # (Later you can return or save these arrays for train-test split)
    


# Correct main guard
if __name__ == "__main__":
    main()
