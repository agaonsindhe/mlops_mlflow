"""
Unit tests for the mlflow_integration module.
"""
from unittest.mock import patch
from src.train_stock_model import train_and_evaluate

@patch("mlflow.log_param")
@patch("mlflow.log_metric")
@patch("mlflow.sklearn.log_model")
def test_mlflow_logging(mock_log_model, mock_log_metric, mock_log_param):
    """
    Test that MLflow logs parameters, metrics, and model correctly.
    """
    # Run the training script
    rmse, r2 = train_and_evaluate(
        data_path="data/stock_data_sample.csv",
        model_path="test_model.pkl"
    )

    # Assert that MLflow logging functions are called
    mock_log_param.assert_called_with("model_type", "Linear Regression")
    mock_log_metric.assert_any_call("rmse", rmse)
    mock_log_metric.assert_any_call("r2", r2)
    mock_log_model.assert_called_once()

