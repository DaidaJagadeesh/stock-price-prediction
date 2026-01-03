# decision_tree_tuned.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# allow imports from src/ (parent of models/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

def create_model_plot_folder():
    this_dir = os.path.dirname(os.path.abspath(__file__))  # src/models/
    plots_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "plots", "models-plots"))
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def run_tuned_decision_tree(x_train, y_train, x_test, y_test, scaler):
    # Ensure y_train 1D for sklearn
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
    dt = DecisionTreeRegressor(random_state=1)

    # NOTE: using 'r2' scoring (regression) instead of "accuracy"
    random_search = RandomizedSearchCV(
        estimator=dt,
        param_distributions=param_grid,
        random_state=0,
        n_iter=100,
        scoring='r2',
        cv=5,
        verbose=True,
        n_jobs=-1
    )

    # Fit search (this may take time)
    print("\nStarting RandomizedSearchCV for DecisionTree (this may run a while)...\n")
    random_search.fit(x_train, y_train_flat)

    print("\nRandomizedSearchCV complete.")
    print("Best params found:", random_search.best_params_)

    # -----------------------
    # Train final model (use best params if available)
    # -----------------------
    best_params = random_search.best_params_
    # build model using best params (fall back to example params if something missing)
    m3 = DecisionTreeRegressor(
        min_samples_split=best_params.get('min_samples_split', 2),
        max_depth=best_params.get('max_depth', 7),
        criterion=best_params.get('criterion', 'squared_error'),
        random_state=1
    )

    m3.fit(x_train, y_train_flat)

    # Predict
    y_pred = m3.predict(x_test)
    print("\ny_pred : ", y_pred)

    # Inverse transform predictions and true values using provided scaler
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(y_pred.shape[0], 1))
    print("\nInverse transform predictions:")
    print(y_pred_inv)

    y_test_in = y_test
    if y_test_in.ndim == 1:
        y_test_in = y_test_in.reshape(y_test_in.shape[0], 1)

    y_test_inv = scaler.inverse_transform(y_test_in)
    print("\nActual values (inverse scaled):")
    print(y_test_inv)

    # Plot actual vs predicted and save
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.legend()
    plt.tight_layout()

    save_folder = create_model_plot_folder()
    save_path = os.path.join(save_folder, "decision_tree_tuned_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"\nDecision Tree (tuned) prediction plot saved at: plots\models-plots\decision_tree_tuned_plot.png\n")

    # accuracy (R2)
    accuracy = r2_score(y_test_inv, y_pred_inv)
    print("Decision Tree (tuned) Accuracy (R2):", accuracy)

    return accuracy, y_pred_inv, y_test_inv

def main():
    # import split utilities
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

    # run tuned decision tree
    run_tuned_decision_tree(x_train, y_train, x_test, y_test, scaler)

if __name__ == "__main__":
    main()
