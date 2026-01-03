# random_forest.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# allow imports from parent folder (src)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------
# Create folder for model plots
# ---------------------------------------------------
def create_model_plot_folder():
    this_dir = os.path.dirname(os.path.abspath(__file__))  # src/models/
    plots_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "plots", "models-plots"))

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    return plots_dir


# ---------------------------------------------------
# Train + Predict Random Forest Model
# ---------------------------------------------------
def run_random_forest(x_train, y_train, x_test, y_test, scaler):

    # Ensure y_train is 1D for sklearn
    if y_train.ndim > 1:
        y_train_flat = y_train.ravel()
    else:
        y_train_flat = y_train

    # Fit Random Forest
    m4 = RandomForestRegressor()
    m4.fit(x_train, y_train_flat)

    # Predict
    y_pred = m4.predict(x_test)
    print("\ny_pred:", y_pred)

    # Inverse transform predictions
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    print("\nInverse transformed predictions:\n", y_pred_inv)

    # Prepare true values for inverse-scaling
    y_test_in = y_test.reshape(-1, 1)
    y_test_inv = scaler.inverse_transform(y_test_in)
    print("\nActual values (inverse scaled):\n", y_test_inv)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.legend()
    plt.tight_layout()

    # Save plot
    save_folder = create_model_plot_folder()
    save_path = os.path.join(save_folder, "random_forest_plot.png")

    plt.savefig(save_path)
    plt.close()

    print(f"\nRandom Forest prediction plot saved at: plots\models-plots\random_forest_plot.png\n")

    # Accuracy (R2 score)
    accuracy = r2_score(y_test_inv, y_pred_inv)
    print("Random Forest Accuracy (R2):", accuracy)

    return accuracy, y_pred_inv, y_test_inv


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    try:
        from split import load_prices, get_yhoo_close, scale_close_stocks, split_close_stocks, train_test_split
    except Exception as e:
        print("Could not import split utilities from split.py:", e)
        print("Make sure split.py exists inside src folder.")
        return

    # Load dataset
    df = load_prices()
    if df is None:
        return

    # Prepare data
    close_stocks = get_yhoo_close(df)
    scaler, scaled_close = scale_close_stocks(close_stocks)
    train, test = split_close_stocks(scaled_close)

    # Create sequences (n_features = 4)
    x_train, y_train, x_test, y_test = train_test_split(train, test, 4)

    # Run Random Forest model
    run_random_forest(x_train, y_train, x_test, y_test, scaler)


if __name__ == "__main__":
    main()
