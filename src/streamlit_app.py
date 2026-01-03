import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# -------------------------------------------------
# Load data safely
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "prices.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df

df = load_data()

# -------------------------------------------------
# Session state for navigation
# -------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
if st.session_state.step == 0:
    st.title("📈 Stock Price Prediction System")

    st.write(
        """
        This application helps users explore stock price data
        and predict the next day's closing price using
        Machine Learning.

        Follow the steps one by one for better understanding.
        """
    )

    if st.button("▶ Start Analysis"):
        st.session_state.step = 1
        st.rerun()

# -------------------------------------------------
# STEP 1: Select Company
# -------------------------------------------------
elif st.session_state.step == 1:
    st.header("Step 1: Select a Company")

    symbols = sorted(df["symbol"].unique())
    selected_symbol = st.selectbox("Choose a stock symbol:", symbols)

    if st.button("Next ▶"):
        st.session_state.selected_symbol = selected_symbol
        st.session_state.step = 2
        st.rerun()

    if st.button("⬅ Back to Home"):
        st.session_state.step = 0
        st.rerun()

# -------------------------------------------------
# STEP 2: View Historical Data
# -------------------------------------------------
elif st.session_state.step == 2:
    st.header("Step 2: View Historical Data")

    company_data = df[df["symbol"] == st.session_state.selected_symbol].sort_values("date")

    st.write(f"Showing first few rows for **{st.session_state.selected_symbol}**")
    st.dataframe(company_data.head())

    if st.button("Next ▶"):
        st.session_state.step = 3
        st.rerun()

    if st.button("⬅ Back to Home"):
        st.session_state.step = 0
        st.rerun()

# -------------------------------------------------
# STEP 3: Visualization
# -------------------------------------------------
elif st.session_state.step == 3:
    st.header("Step 3: Price Visualization")

    company_data = df[df["symbol"] == st.session_state.selected_symbol].sort_values("date")

    st.subheader("Closing Price Over Time")
    st.line_chart(company_data.set_index("date")["close"])

    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=company_data["date"],
        open=company_data["open"],
        high=company_data["high"],
        low=company_data["low"],
        close=company_data["close"]
    )])
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Next ▶"):
        st.session_state.step = 4
        st.rerun()

    if st.button("⬅ Back to Home"):
        st.session_state.step = 0
        st.rerun()

# -------------------------------------------------
# STEP 4: Train Machine Learning Model
# -------------------------------------------------
elif st.session_state.step == 4:
    st.header("Step 4: Train Machine Learning Model")

    company_data = df[df["symbol"] == st.session_state.selected_symbol].sort_values("date")

    close_prices = company_data["close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(close_prices)

    train_size = int(len(scaled_prices) * 0.8)
    train = scaled_prices[:train_size]
    test = scaled_prices[train_size:]

    n_features = 4
    X_train, y_train = [], []
    X_test, y_test = [], []

    for i in range(n_features, len(train)):
        X_train.append(train[i - n_features:i, 0])
        y_train.append(train[i, 0])

    for i in range(n_features, len(test)):
        X_test.append(test[i - n_features:i, 0])
        y_test.append(test[i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    accuracy = r2_score(y_test_inv, y_pred_inv)

    st.write(f"Model R² Score: **{accuracy:.4f}**")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=y_test_inv.flatten(), name="Actual"))
    fig2.add_trace(go.Scatter(y=y_pred_inv.flatten(), name="Predicted"))
    st.plotly_chart(fig2, use_container_width=True)

    st.session_state.scaler = scaler
    st.session_state.model = model
    st.session_state.scaled_prices = scaled_prices

    if st.button("Next ▶"):
        st.session_state.step = 5
        st.rerun()

    if st.button("⬅ Back to Home"):
        st.session_state.step = 0
        st.rerun()

# -------------------------------------------------
# STEP 5: Next-Day Prediction
# -------------------------------------------------
elif st.session_state.step == 5:
    st.header("Step 5: Next-Day Price Prediction")

    scaler = st.session_state.scaler
    model = st.session_state.model
    scaled_prices = st.session_state.scaled_prices

    n_features = 4
    next_input = scaled_prices[-n_features:].reshape(1, -1)
    next_scaled_price = model.predict(next_input)
    next_price = scaler.inverse_transform(next_scaled_price.reshape(-1, 1))[0][0]

    st.success(
        f"Predicted next-day closing price for "
        f"**{st.session_state.selected_symbol}**: ₹ {next_price:.2f}"
    )

    if st.button("🏠 Back to Home"):
        st.session_state.step = 0
        st.rerun()
