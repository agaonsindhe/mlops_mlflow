"""
This module contains utility functions for the stock model project.
"""
import pandas as pd

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

def add_features(data):
    """
    Add new features to the dataset for better predictions.

    Args:
        data (pd.DataFrame): Original dataset.

    Returns:
        pd.DataFrame: Dataset with new features.
    """
    data['Close_ma_3'] = data['Close'].rolling(window=3).mean()
    data['Close_ma_7'] = data['Close'].rolling(window=7).mean()
    data['Close_lag_1'] = data['Close'].shift(1)
    data['Close_pct_change'] = data['Close'].pct_change()

    data.dropna(inplace=True)

    return data
