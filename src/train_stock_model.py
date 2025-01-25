"""
This module contains functions to train and evaluate a Linear Regression model
for stock price prediction using historical data.
"""
import subprocess
import sys
import time
from math import sqrt
import pickle
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from src.utils import add_features, load_config, get_dataset_path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="mlflow.gateway.config")

def ensure_dvc_data():
    """
    Ensure all DVC-tracked files are pulled before running the training.
    """
    try:
        print("Pulling DVC data...")
        subprocess.run(["dvc", "pull"], check=True)
        print("DVC data is up to date.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull DVC data: {e}")
        sys.exit(1)

def load_data(data_path):
    """
    Load stock market data from a CSV file.

    Args:
        data_path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', inplace=True)
    return data


def train_and_evaluate(config_path="config.yaml", model_path="model.pkl"):
    """
    Train and evaluate a Linear Regression model.

    Args:
        config_path (str): Path to the configuration file.
        model_path (str): Path to save the trained model.

    Returns:
        tuple: (rmse, r2, mae, training_time, evs)
    """
    # Load configuration
    config = load_config(config_path)

    # Get the dataset path dynamically
    data_path = get_dataset_path(config)

    # Start an MLflow experiment
    mlflow.set_experiment("Stock Price Prediction")

    with mlflow.start_run():
        mlflow.log_param("model_type", "Ridge Regression")
        mlflow.log_param("dataset_size", "200 MB")
        mlflow.log_param("features_engineered", True)
        mlflow.log_param("hyperparameter_tuning", True)
        mlflow.log_param("tuning_method", "GridSearchCV")

    # Start timing
    start_time = time.time()


    # Load and preprocess data
    data = load_data(data_path)
    data = add_features(data)
    features = ['Open', 'High', 'Low', 'Volume', 'Close_ma_3', 'Close_ma_7', 'Close_lag_1', 'Close_pct_change']
    target = 'Close'

    # Split the dataset
    x_train, x_test, y_train, y_test = split_data(data, features, target)

    # Define model and expanded parameter grid
    model = Ridge()
    param_grid = {
       "alpha": [0.01, 0.1, 1.0],
       "solver": ["lsqr", "svd", "cholesky"],
       "max_iter": [50, 50, 50],
       "tol": [1e-4, 1e-3, 1e-2],
    }

    # Perform GridSearchCV
    random_search = RandomizedSearchCV(model, param_grid, n_iter=50,scoring="neg_mean_squared_error", cv=2, verbose=1,random_state=42)
    random_search.fit(x_train, y_train)

    # Log results of all hyperparameter combinations
    cv_results = random_search.cv_results_
    for i in range(len(cv_results["params"])):
        with mlflow.start_run(nested=True):  # Nested runs for each combination
            # Log the hyperparameters for this combination
            mlflow.log_params(cv_results["params"][i])

            # Log mean and std of scores
            mean_test_score = cv_results["mean_test_score"][i]
            std_test_score = cv_results["std_test_score"][i]

            mlflow.log_metric("mean_test_score", -mean_test_score)  # Convert to positive
            mlflow.log_metric("std_test_score", std_test_score)

            # Optionally, log fold-wise scores
            for j in range(random_search.cv):
                mlflow.log_metric(f"split_{j}_test_score", -cv_results[f"split{j}_test_score"][i])

    # Log the best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Stop timing
    training_time = time.time() - start_time


    # Evaluate the model
    y_pred = best_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    with mlflow.start_run(nested=True):  # Log the final best model as a separate run
        # Log metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("explained_variance", evs)
        mlflow.sklearn.log_model(best_model, "model")

    print(f"Model RMSE: {rmse:.2f}")
    print(f"Model MAE: {mae:.2f}")
    print(f"Model R²: {r2:.2f}")
    print(f"Training Time: {training_time:.2f} seconds")

    # Save the model
    save_model(best_model, model_path)
    # Log training time
    total_time = time.time() - start_time
    mlflow.log_metric("training_time", total_time)

    return rmse, r2,mae, training_time, evs

def save_model(model, model_path):
    """
    Saves the trained model to a file.

    Args:
        model: Trained model object.
        model_path (str): Path to save the model.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to '{model_path}'")

def split_data(data, features, target):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset.
        features (list): List of feature column names.
        target (str): Target column name.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    features_data = data[features]
    target_data = data[target]
    return train_test_split(features_data, target_data, test_size=0.2, random_state=42)


if __name__ == "__main__":
    # Ensure DVC data is available
    ensure_dvc_data()

    # Run training and evaluation
    train_and_evaluate()
