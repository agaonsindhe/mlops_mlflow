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
