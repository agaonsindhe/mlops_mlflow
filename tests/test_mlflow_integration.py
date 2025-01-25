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
    rmse, r2, mae, training_time, evs= train_and_evaluate(
        config_path="config.yaml",
        model_path="model.pkl"
    )

    # Assert that MLflow logging functions are called for parameters
    mock_log_param.assert_any_call("model_type", "Linear Regression")
    mock_log_param.assert_any_call("learning_rate", 0.01)  # Example hyperparameter
    mock_log_param.assert_any_call("batch_size", 32)  # Example hyperparameter

    # Assert that MLflow logging functions are called for metrics
    mock_log_metric.assert_any_call("rmse", rmse)
    mock_log_metric.assert_any_call("r2", r2)
    mock_log_metric.assert_any_call("mae", mae)
    mock_log_metric.assert_any_call("training_time", training_time)
    mock_log_metric.assert_any_call("explained_variance", evs)

    # Assert that the model is logged as an artifact
    mock_log_model.assert_called_once()
