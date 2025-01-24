"""
Unit tests for the utils module.
"""

from src.utils import load_data

def test_load_data():
    """
    Test the load_data function to ensure it loads the dataset correctly.
    """
    data = load_data('data/stock_data_sample.csv')
    assert not data.empty, "Data should not be empty"
    assert 'Date' in data.columns, "Data should contain 'Date' column"
