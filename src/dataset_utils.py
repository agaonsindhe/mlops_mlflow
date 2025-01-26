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

def get_dataset_version(dataset_path):
    """
    Get dataset version from a CSV file.
    :param dataset_path:
    :return:
    """
    from dvc.repo import Repo

    # Initialize the DVC repository
    repo = Repo()

    # Get the hash of the dataset
    dataset_info = repo.find_outs_by_path(dataset_path)[0]
    dataset_hash = dataset_info.hash_info.value

    print(f"Dataset hash: {dataset_hash}")
    return dataset_hash