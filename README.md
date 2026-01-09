# Stock Price Prediction System

This project is a stock price prediction system built using Python and Machine Learning.  
It allows users to explore historical stock price data, visualize price trends, and predict the next day’s closing price.  
A simple Streamlit web application is included so that users can interact with the system easily.

---

## Project Description

Stock prices change continuously and depend on multiple factors.  
In this project, historical stock price data is used to understand trends and make predictions using machine learning models.

The project is implemented in a step-by-step manner:
- Data exploration and visualization
- Data preprocessing and scaling
- Train-test splitting for time-series data
- Model training and evaluation
- Prediction using a Streamlit web interface

The main goal of this project is clarity, simplicity, and correct implementation of the machine learning workflow.

---

## Folder Structure
```
Stock_price_prediction/
│
├── dataset/
│ ├── prices.csv
│ └── securities.csv
│
├── src/
│ ├── train.py # Data exploration and plotting
│ ├── prepare.py # Data filtering and scaling
│ ├── split.py # Train-test split and sequence creation
│ ├── streamlit_app.py # Streamlit user interface
│ │
│ └── models/
│ ├── linear_regression.py
│ ├── decision_tree.py
│ ├── random_forest.py
│
├── plots/
│ └── Saved stock price plots
│
├── .venv/ # Virtual environment
├── requirements.txt # Required libraries
├── README.md

```

---

## Dataset Information

### prices.csv
This file contains daily stock price data for multiple companies.  
Important columns include:
- date  
- symbol  
- open  
- high  
- low  
- close  
- volume  

### securities.csv
This file contains company-related details such as:
- Company name
- Ticker symbol
- Sector
- Industry information

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Plotly  
- Scikit-learn  
- Streamlit  

---

## Machine Learning Approach

Multiple machine learning models were implemented in this project to predict stock prices.

### 1. Linear Regression
- Used as the baseline model.
- Simple and easy to understand.
- Helps explain the relationship between past prices and future prices.

### 2. Decision Tree Regressor
- Used to capture non-linear patterns in stock prices.
- Performs better than linear models in some cases.
- Hyperparameter tuning was applied to improve performance.
- Tuned parameters include:
  - `max_depth`
  - `min_samples_split`
  - `criterion`

### 3. Random Forest Regressor
- An ensemble model that combines multiple decision trees.
- Helps reduce overfitting and improves prediction stability.
- Hyperparameter tuning was performed using randomized search.
- Tuned parameters include:
  - `max_depth`
  - `min_samples_split`
  - `criterion`

### Feature Details
- Uses the last **4 days of closing prices** as input features.
- Target variable is the **next day’s closing price**.

### Data Handling
- Data split: **80% training**, **20% testing**
- Feature scaling using **MinMaxScaler**
- Evaluation metric: **R² score**

Each company is modeled separately to avoid mixing unrelated stock patterns.

---

## Streamlit Application Workflow

The Streamlit application follows a clear and guided step-by-step process:

1. **Home Page**  
   Introduction to the application and option to start analysis.

2. **Select Company**  
   User selects a stock symbol from a dropdown list.

3. **View Historical Data**  
   Displays sample rows of the selected company’s stock data.

4. **Price Visualization**  
   - Line chart of closing prices  
   - Candlestick chart showing open, high, low, and close prices  

5. **Train Machine Learning Model**  
   - Model is trained using historical data  
   - R² score is displayed  
   - Actual vs Predicted price graph is shown  

6. **Next-Day Prediction**  
   - Predicts the next day’s closing price  
   - User can return to the home page and start again  

This step-by-step design makes the application easy to use for non-technical users.

---

## How to Run the Project

### Step 1: Activate Virtual Environment
```bash
.venv\Scripts\activate
Step 2: Install Dependencies

bash
Copy code
pip install -r requirements.txt
Step 3: (Optional) Run Data Exploration Script
bash
Copy code
python src/train.py
Step 4: Run Streamlit Application
bash
Copy code
streamlit run src/streamlit_app.py
The application will open in the browser.

Key Features
Step-by-step guided user interface

Interactive stock price charts

Multiple machine learning models

Hyperparameter tuning for better performance

Clear visualization of predictions

Easy restart and navigation

Limitations:

Uses only historical price data

External factors such as news and market sentiment are not included

Predicts only the next day’s closing price

FutureImprovements:

Add deep learning models like LSTM

Predict prices for multiple future days

Deploy the application online

Conclusion
This project demonstrates a complete machine learning pipeline, starting from data exploration and preprocessing to model training and prediction.

The use of multiple models and hyperparameter tuning improves understanding of model performance, while the Streamlit interface makes the system easy to use and understand.


