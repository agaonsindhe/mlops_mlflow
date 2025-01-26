"""
The module contains additional features for stock prize prediction
"""
import pandas as pd

def add_features_to_dataset(file_path):
    """
    Add new features to the dataset and overwrite the existing file.

    Args:
        file_path (str): Path to the dataset file.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Add new features
    df['Close_ma_3'] = df['Close'].rolling(window=3).mean()
    df['Close_ma_7'] = df['Close'].rolling(window=7).mean()
    df['Close_lag_1'] = df['Close'].shift(1)  # Previous day's close price

    # Save the updated dataset (overwrite the file)
    df.to_csv(file_path, index=False)
    print(f"Updated dataset written to: {file_path}")


if __name__ == "__main__":
    # Update the dataset in place
    DATASET_PATH = "data/stocks_df.csv"
    add_features_to_dataset(DATASET_PATH)
