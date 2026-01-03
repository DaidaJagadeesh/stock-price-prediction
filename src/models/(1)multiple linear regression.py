# model_linear.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# allow imports from src/ (parent of models/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_model_plot_folder():
    this_dir = os.path.dirname(os.path.abspath(__file__))  # src/models/
    plots_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "plots", "models-plots"))
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def run_linear_model(x_train, y_train, x_test, y_test, scaler):
    if y_train.ndim > 1:
        y_train_flat = y_train.ravel()
    else:
        y_train_flat = y_train
    m1 = LinearRegression()
    m1.fit(x_train, y_train_flat)
    y_pred = m1.predict(x_test)
    print("\ny_pred : ", y_pred)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(y_pred.shape[0], 1))
    print("\nInverse transform predictions and true values using provided scaler")
    print(y_pred_inv)
    y_test_in = y_test
    if y_test_in.ndim == 1:
        y_test_in = y_test_in.reshape(y_test_in.shape[0], 1)
    y_test_inv = scaler.inverse_transform(y_test_in)
    print(y_test_inv)
    # Plot actual vs predicted and save to models-plots
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.legend()
    plt.tight_layout()
    save_folder = create_model_plot_folder()
    save_path = os.path.join(save_folder, "Multiple_linear_regression_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"\nModel prediction plot saved at: plots\models-plots\Multiple_linear_regression_plot.png\n")
    accuracy = r2_score(y_test_inv, y_pred_inv)
    print("accuracy : ", accuracy)
    return accuracy, y_pred_inv, y_test_inv

def main():
    try:
        from split import load_prices, get_yhoo_close, scale_close_stocks, split_close_stocks, train_test_split
    except Exception as e:
        print("Could not import split utilities from split.py:", e)
        print("Ensure split.py exists in src/ and is runnable.")
        return
    df = load_prices()
    if df is None:
        return
    close_stocks = get_yhoo_close(df)
    scaler, scaled_close = scale_close_stocks(close_stocks)
    train, test = split_close_stocks(scaled_close)
    x_train, y_train, x_test, y_test = train_test_split(train, test, 4)
    run_linear_model(x_train, y_train, x_test, y_test, scaler)

if __name__ == "__main__":
    main()
