"""
Unit tests for the train_stock_model module.
"""

from src.train_stock_model import train_and_evaluate

def test_train_and_evaluate():
    """
    Test the train_and_evaluate function for expected outputs.
    """
    rmse, r2 = train_and_evaluate(
        data_path='data/stock_data_sample.csv',
        model_path='test_model.pkl'
    )
    assert rmse > 0, "RMSE should be greater than 0"
    assert 0 <= r2 <= 1, "R² should be between 0 and 1"
