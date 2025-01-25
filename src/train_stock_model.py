"""
This module contains functions to train and evaluate a Linear Regression model
for stock price prediction using historical data.
"""
from math import sqrt

from src.dataset_utils import load_data, add_features
from src.utils import load_config, get_config_path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import mlflow
import pickle



def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on test data and return metrics.
    """
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"rmse": rmse, "mae": mae, "evs": evs, "r2": r2}

def train_and_log_runs(config_path="config.yaml",):
    """
    Train and log three different Ridge regression models with varying hyperparameters and log the best model.
    """
    # Start MLflow Experiment
    mlflow.set_experiment("Stock Price Prediction - Multiple Runs")
    # Load configuration
    config = load_config(config_path)

    # Get the dataset path dynamically
    data_path, model_path = get_config_path(config)

    # Load and preprocess data
    df = load_data(data_path)
    df = add_features(df)

    # Dynamically select features based on existing columns
    required_features = ['Open', 'High', 'Low', 'Volume', 'Close_ma_3', 'Close_ma_7', 'Close_lag_1']
    features = [col for col in required_features if col in df.columns]

    if not features:
        raise ValueError("No valid features found in the dataset. Check the dataset for missing columns.")

    target = 'Close'
    features = df[features].dropna()
    target = df[target].dropna()
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Define hyperparameters for three runs
    runs = [
        {"alpha": 0.1, "solver": "auto"},
        {"alpha": 1.0, "solver": "svd"},
        {"alpha": 10.0, "solver": "cholesky"},
    ]

    best_model = None
    best_metrics = None
    best_run_id = None

    # Run experiments
    for i, params in enumerate(runs):
        with mlflow.start_run():
            # Initialize model with current parameters
            model = Ridge(alpha=params["alpha"], solver=params["solver"])
            model.fit(x_train, y_train)

            # Evaluate model
            metrics = evaluate_model(model, x_test, y_test)

            # Log parameters and metrics
            mlflow.log_param("run_index", i + 1)
            mlflow.log_param("alpha", params["alpha"])
            mlflow.log_param("solver", params["solver"])
            mlflow.log_param("dataset_version", "v1")
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("evs", metrics["evs"])
            mlflow.log_metric("r2", metrics["r2"])

            # Log the trained model
            mlflow.sklearn.log_model(model, f"model_run_{i + 1}")

            # Update the best model if applicable
            if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
                best_model = model
                best_metrics = metrics
                best_run_id = mlflow.active_run().info.run_id

    # Log the best model as a separate artifact
    if best_model:
        with mlflow.start_run(run_id=best_run_id, nested=True):
            mlflow.log_param("best_model", True)
            mlflow.log_metric("best_rmse", best_metrics["rmse"])
            mlflow.log_metric("best_mae", best_metrics["mae"])
            mlflow.log_metric("best_evs", best_metrics["evs"])
            mlflow.log_metric("best_r2", best_metrics["r2"])
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            mlflow.sklearn.log_model(best_model, "best_model")

    print(f"Best model saved to {model_path}")
    print(f"Best metrics: {best_metrics}")

if __name__ == "__main__":

    # Run training and evaluation
    train_and_log_runs("config.yaml")
