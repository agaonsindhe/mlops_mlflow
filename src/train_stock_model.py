"""
This module contains functions to train and evaluate a Linear Regression model
for stock price prediction using historical data.
"""
from math import sqrt
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import add_features

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


def train_and_evaluate(
        data_path='data/stock_data_sample.csv',
        model_path='model.pkl'
):
    """
    Train and evaluate a Linear Regression model.

    Args:
        data_path (str): Path to the dataset.
        model_path (str): Path to save the trained model.

    Returns:
        tuple: RMSE and R² scores.
    """

    # Start an MLflow experiment
    mlflow.set_experiment("Stock Price Prediction")

    with mlflow.start_run():
    # Load and preprocess data
        data = load_data(data_path)
        data = add_features(data)
        features = ['Open', 'High', 'Low', 'Volume', 'Close_ma_3', 'Close_ma_7', 'Close_lag_1', 'Close_pct_change']
        target = 'Close'

        # Split the dataset
        x_train, x_test, y_train, y_test = split_data(data, features, target)

        # Train the model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Evaluate the model
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model RMSE: {rmse:.2f}")
        print(f"Model R²: {r2:.2f}")

        # Log parameters, metrics, and model
        mlflow.log_param("features", features)
        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log the model as an artifact
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model RMSE: {rmse:.2f}")
        print(f"Model R²: {r2:.2f}")
        print(f"Model saved to '{model_path}'")

        # Save the model
        save_model(model, model_path)

    return rmse, r2

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
    train_and_evaluate()
