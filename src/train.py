# ==========================================
# IMPORTS (required libraries)
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===========================
# Step 1: Load the CSV
# ===========================

def load_prices():
    path = os.path.join("..", "dataset", "prices.csv")
    print("Loaded", path, "\n")

    if not os.path.exists(path):
        print("Error: file not found:", path)
        return None

    df = pd.read_csv(path)

    # If you really want to print the full DataFrame, keep the next line.
    # Often df.head() is more readable:
    print(df, "\n")

    print("shape : ",df.shape)
    print("Columns:",df.columns)

    # show unique symbols (print instead of display)
    if "symbol" in df.columns:
        unique_symbols = df["symbol"].unique()
        print("unique symbols (count):", unique_symbols.shape[0])
        print("unique symbols:", unique_symbols, "\n")
    else:
        print("No 'symbol' column found in DataFrame.")

    # info() prints to stdout already
    df.info()
    print("\nchecking for null values: ")
    print(df.isnull().sum(),"\n")

    return df

def load_company_info():
    
    path = os.path.join("..", "dataset", "securities.csv")
    print("Loaded", path, "\n")

    if not os.path.exists(path):
        print("Error: securities.csv not found at", path)
        return None

    comp_info = pd.read_csv(path)
    print(comp_info)
    
    print("Unique Ticker symbols:",comp_info["Ticker symbol"].nunique(),"\n")
    print("In price.csv we had 501 unique stocks but there are a total of 505 stocks in securites.csv\n")
    
    comp_info.info()
    
    print("\nchecking for null values:")
    print(comp_info.isnull().sum())
    print("\nWe  will take some companies stock data and check weather they are related\n")
    
    return comp_info

def get_selected_symbols(comp_info):
    selected_companies = [
        "Yahoo Inc.",
        "Xerox Corp.",
        "Adobe Systems Inc",
        "Microsoft Corp.",
        "Facebook",
        "Goldman Sachs Group"
    ]

    comp_sym = comp_info.loc[
        comp_info["Security"].isin(selected_companies),
        "Ticker symbol"
    ]
    
    print(comp_sym)
    print("\nSelected symbols:")
    for sym in comp_sym:
        print(sym)

    return comp_sym

def create_plot_folder(folder_name="plots"):
    """
    Create a plots folder at project root (one level up from src/)
    Returns the absolute folder path.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.normpath(os.path.join(this_dir, "..", folder_name))
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder



def plotter(sym, df, save_folder):
    # Filter once
    company = df[df["symbol"] == sym]

    if company.empty:
        print(f"No data found for symbol: {sym}")
        return

    # Extract open & close values
    company_open = company["open"].astype("float32").values
    company_close = company["close"].astype("float32").values

    # Create figure
    plt.figure(figsize=(16, 8))

    # OPEN plot
    plt.subplot(211)
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel(f"{sym} Open Price")
    plt.title(f"{sym} - Open Price vs Time")
    plt.plot(company_open, "g")

    # CLOSE plot
    plt.subplot(212)
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel(f"{sym} Close Price")
    plt.title(f"{sym} - Close Price vs Time")
    plt.plot(company_close, "b")

    plt.tight_layout()

    # Save to file
    save_path = os.path.join(save_folder, f"{sym}_plot.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot for {sym} → {save_path}")

def plot_selected_symbols(df, comp_sym):
    save_folder = create_plot_folder()
    

    for sym in comp_sym:
        plotter(sym, df, save_folder)
        
    print("\nFrom the above plots (saved in the 'plots' folder), we can confirm that no two companies' stocks are dependent on each other. "
      "This means we cannot build a single model using the entire dataset; "
      "instead, we must build separate models for each company using only that company's data.\n")
    print("\nData exploration and plotting are complete\n")
    print("Next step: Run 'prepare.py' to clean the data and generate features.\n")


# ===========================
# Main function
# ===========================

def main():
    df = load_prices()
    comp_info = load_company_info()
    selected_symbols = get_selected_symbols(comp_info)
    plot_selected_symbols(df, selected_symbols)
    # further steps...

if __name__ == "__main__":
    main()
