"""
This module contains utility functions for the stock model project.
"""
import os
import glob
import pandas as pd
import yaml
from azure.storage.blob import BlobServiceClient

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

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}") from e

def get_dataset_path(config):
    """
    Dynamically determine the dataset path based on the config.

    Args:
        config (dict): Configuration dictionary with preferred and fallback paths.

    Returns:
        str: Path to the selected dataset.

    Raises:
        FileNotFoundError: If no valid dataset is found.
    """
    preferred_path = config["dataset"]["preferred_path"]
    fallback_path = config["dataset"]["fallback_path"]

    # Check if preferred file exists
    if os.path.exists(preferred_path):
        print(f"Using preferred dataset: {preferred_path}")
        return preferred_path

    # Check if fallback file exists
    if os.path.exists(fallback_path):
        print(f"Using fallback dataset: {fallback_path}")
        return fallback_path

    # Handle case for multiple files in directory
    dataset_dir = os.path.dirname(preferred_path)
    files = glob.glob(f"{dataset_dir}/*.csv")
    if files:
        print(f"Multiple files found, selecting first: {files[0]}")
        return files[0]

    raise FileNotFoundError("No valid dataset found.")

def list_blob_files(container_name, connection_string):
    """
    List all files in an Azure Blob Storage container.

    Args:
        container_name (str): Name of the blob container.
        connection_string (str): Azure Blob connection string.

    Returns:
        list: List of file names in the container.
    """
    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)

    files = [blob.name for blob in container_client.list_blobs()]
    return files

def get_dataset_from_blob(container_name, connection_string, preferred_filename):
    """
    Get the dataset file from Azure Blob Storage.

    Args:
        container_name (str): Blob container name.
        connection_string (str): Azure Blob connection string.
        preferred_filename (str): Preferred dataset file name.

    Returns:
        str: Path to the downloaded dataset.
    """
    files = list_blob_files(container_name, connection_string)
    if preferred_filename in files:
        print(f"Found preferred file: {preferred_filename}")
        # Download logic here
    else:
        print("Preferred file not found. Selecting first file in container.")
        # Download the first file
