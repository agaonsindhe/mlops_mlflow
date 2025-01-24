"""
This module contains functions to train and evaluate a Linear Regression model
for stock price prediction using historical data.
"""
from math import sqrt
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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
    # Load and prepare data
    data = load_data(data_path)
    features = ['Open', 'High', 'Low', 'Volume']
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
    r2 = r2_score(y_test, y_pred)

    print(f"Model RMSE: {rmse:.2f}")
    print(f"Model R²: {r2:.2f}")

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
