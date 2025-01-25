"""
This module contains dataset utility functions for the stock model project.
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
    Add new features to the dataset for better predictions, only if the required columns exist.

    Args:
        data (pd.DataFrame): Original dataset.

    Returns:
        pd.DataFrame: Dataset with new features.
    """
    # Check if the 'Close' column exists before adding features
    if 'Close' in data.columns:
        # Add rolling mean features
        if 'Close_ma_3' not in data.columns:
            data['Close_ma_3'] = data['Close'].rolling(window=3).mean()

        if 'Close_ma_7' not in data.columns:
            data['Close_ma_7'] = data['Close'].rolling(window=7).mean()

        # Add lag feature
        if 'Close_lag_1' not in data.columns:
            data['Close_lag_1'] = data['Close'].shift(1)

        # Add percentage change
        if 'Close_pct_change' not in data.columns:
            data['Close_pct_change'] = data['Close'].pct_change()

        # Drop rows with NaN values resulting from rolling and shifting
        data.dropna(inplace=True)

    else:
        raise ValueError("The required column 'Close' is missing from the dataset.")

    return data
