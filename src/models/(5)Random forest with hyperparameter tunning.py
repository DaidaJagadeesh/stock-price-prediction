# random_forest_tuned.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# allow imports from src/ (parent of models/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

def create_model_plot_folder():
    this_dir = os.path.dirname(os.path.abspath(__file__))  # src/models/
    plots_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "plots", "models-plots"))
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def run_tuned_random_forest(x_train, y_train, x_test, y_test, scaler, n_iter=100):
    # ensure y_train is 1D
    if y_train.ndim > 1:
        y_train_flat = y_train.ravel()
    else:
        y_train_flat = y_train

    # -----------------------
    # Hyperparameter search
    # -----------------------
    param_grid = {
        'criterion': ['squared_error', 'absolute_error'],
        'min_samples_split': range(2, 10),
        'max_depth': range(2, 10)
    }

    rf = RandomForestRegressor(random_state=1)

    # NOTE: use 'r2' scoring for regression
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        random_state=0,
        n_iter=n_iter,
        scoring='r2',
        cv=5,
        verbose=True,
        n_jobs=-1
    )

    print("\nStarting RandomizedSearchCV for RandomForest (this may run a while)...\n")
    random_search.fit(x_train, y_train_flat)

    print("\nRandomizedSearchCV complete.")
    print("Best params found:", random_search.best_params_)

    # -----------------------
    # Train final model using best params
    # -----------------------
    best_params = random_search.best_params_
    m5 = RandomForestRegressor(
        min_samples_split=best_params.get('min_samples_split', 2),
        max_depth=best_params.get('max_depth', 7),
        criterion=best_params.get('criterion', 'squared_error'),
        random_state=1,
        n_jobs=-1
    )

    m5.fit(x_train, y_train_flat)

    # Predict
    y_pred = m5.predict(x_test)
    print("\ny_pred : ", y_pred)

    # Inverse transform predictions
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(y_pred.shape[0], 1))
    print("\nInverse transform predictions:")
    print(y_pred_inv)

    # Inverse transform actuals
    y_test_in = y_test
    if y_test_in.ndim == 1:
        y_test_in = y_test_in.reshape(y_test_in.shape[0], 1)
    y_test_inv = scaler.inverse_transform(y_test_in)
    print("\nActual values (inverse scaled):")
    print(y_test_inv)

    # Plot and save
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.legend()
    plt.tight_layout()

    save_folder = create_model_plot_folder()
    save_path = os.path.join(save_folder, "random_forest_tuned_plot.png")
    plt.savefig(save_path)
    plt.close()

    print(f"\nRandom Forest (tuned) prediction plot saved at: plots\models-plots\random_forest_tuned_plot.png\n")

    # accuracy (R2)
    accuracy = r2_score(y_test_inv, y_pred_inv)
    print("Random Forest (tuned) Accuracy (R2):", accuracy)

    return accuracy, y_pred_inv, y_test_inv

def main():
    try:
        from split import load_prices, get_yhoo_close, scale_close_stocks, split_close_stocks, train_test_split
    except Exception as e:
        print("Could not import split utilities from split.py:", e)
        print("Ensure split.py exists in src/ and is runnable.")
        return

    # Prepare data
    df = load_prices()
    if df is None:
        return

    close_stocks = get_yhoo_close(df)
    scaler, scaled_close = scale_close_stocks(close_stocks)
    train, test = split_close_stocks(scaled_close)

    # create sequences (n_features = 4)
    x_train, y_train, x_test, y_test = train_test_split(train, test, 4)

    # run tuned random forest (n_iter stays 100 unless you pass smaller)
    run_tuned_random_forest(x_train, y_train, x_test, y_test, scaler, n_iter=100)

if __name__ == "__main__":
    main()
