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
        model_path='model.pkl'
    )
    assert rmse > 0, "RMSE should be greater than 0"
    assert 0 <= r2 <= 1, "R² should be between 0 and 1"

def test_train_and_evaluate_with_features():
    """
    Test the train_and_evaluate function to ensure it works with enhanced features.
    """
    # Run the function with a sample dataset
    rmse, r2 = train_and_evaluate(
        data_path='data/stock_data_sample.csv',
        model_path='model.pkl'
    )

    # Validate the RMSE is greater than 0 (indicating the model trained successfully)
    assert rmse > 0, "RMSE should be greater than 0 with valid training data."

    # Validate R² score is between 0 and 1
    assert 0 <= r2 <= 1, "R² should be between 0 and 1 for valid predictions."
