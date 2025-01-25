"""
Unit tests for the train_stock_model module.
"""

from src.train_stock_model import train_and_evaluate


def test_train_and_evaluate():
    """
    Test the train_and_evaluate function for dynamic dataset paths.
    """

    rmse, r2,mae, training_time, evs = train_and_evaluate(
        config_path="config.yaml",
        model_path='model.pkl'
    )

    # Validate RMSE is positive
    assert rmse > 0, "RMSE should be greater than 0"

    # Validate R² is between 0 and 1
    assert 0 <= r2 <= 1, "R² should be between 0 and 1"

    # Validate MAE is positive
    assert mae > 0, "MAE should be greater than 0"

    # Validate training time is positive
    assert training_time > 0, "Training time should be greater than 0"

    # Validate EVS is between 0 and 1
    assert 0 <= evs <= 1, "Explained Variance Score should be between 0 and 1"

def test_train_and_evaluate_with_features():
    """
    Test the train_and_evaluate function to ensure it works with enhanced features.
    """
    # Run the function with a sample dataset
    rmse, r2, mae, training_time, evs = train_and_evaluate(
        config_path='config.yaml',
        model_path='model.pkl'
    )

    # Validate RMSE is positive
    assert rmse > 0, "RMSE should be greater than 0"

    # Validate R² is between 0 and 1
    assert 0 <= r2 <= 1, "R² should be between 0 and 1"

    # Validate MAE is positive
    assert mae > 0, "MAE should be greater than 0"

    # Validate training time is positive
    assert training_time > 0, "Training time should be greater than 0"

    # Validate EVS is between 0 and 1
    assert 0 <= evs <= 1, "Explained Variance Score should be between 0 and 1"
