"""
This is test module for dataset utils.
"""

import pandas as pd
from src.dataset_utils import load_data


def test_load_data(mocker):
    """
    Test the load_data function to ensure it loads and processes the dataset correctly.
    """
    mock_data = {
        'Date': ['2025-01-01', '2025-01-03', '2025-01-02'],
        'Open': [100, 101, 102],
        'High': [105, 106, 107],
        'Low': [95, 96, 97],
        'Close': [102, 103, 104],
        'Volume': [1000, 1100, 1200],
    }
    mock_df = pd.DataFrame(mock_data)

    expected_df = mock_df.copy()
    expected_df['Date'] = pd.to_datetime(expected_df['Date'])
    expected_df.sort_values(by='Date', inplace=True)

    mocker.patch("pandas.read_csv", return_value=mock_df)

    df = load_data("dummy_path.csv")

    assert isinstance(df, pd.DataFrame)
    assert df.equals(expected_df)


