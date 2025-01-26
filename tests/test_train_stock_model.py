"""
This is unit test for train_stock_model.py
"""
from mlflow import start_run, log_param, sklearn, log_metric
import pandas as pd
from src.train_stock_model import train_and_log_runs


def test_train_and_log_runs_v1_dataset(mocker):
    """
    Test train_and_log_runs with version 1 of the dataset (missing some feature columns).
    """
    # Mock configuration and paths
    mocker.patch("src.train_stock_model.load_config", return_value={"some_config_key": "some_value"})
    mocker.patch("src.train_stock_model.get_config_path", return_value=("test_data_path.csv", "test_model_path.pkl"))

    # Mock version 1 dataset
    data_v1 = pd.DataFrame({
        "Open": [1, 2, 3, 4, 5, 6],
        "High": [2, 3, 4, 5, 6, 7],
        "Low": [0, 1, 2, 3, 4, 5],
        "Volume": [100, 200, 300, 400, 500, 600],
        "Close": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
    })
    data_v1['Close_pct_change'] = data_v1['Close'].pct_change()

    mocker.patch("src.train_stock_model.load_data", return_value=data_v1)

    # Mock MLflow start_run to return a mock object with an info.run_id attribute
    mock_run = mocker.MagicMock()
    mock_run.info.run_id = "test_run_id"
    mocker.patch("mlflow.start_run", return_value=mock_run)

    # Mock other MLflow functions
    mocker.patch("mlflow.log_param")
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.sklearn.log_model")
    mocker.patch("mlflow.set_experiment")

    # Run the function
    train_and_log_runs("mock_config.yaml")

    # Validate that MLflow start_run was called
    start_run.assert_called()
    log_param.assert_called()
    log_metric.assert_called()
    sklearn.log_model.assert_called()

# def test_train_and_log_runs_v2_dataset(mocker):
#     """
#     Test train_and_log_runs with version 2 of the dataset (all feature columns already exist).
#     """
#     # Mock configuration and paths
#     mocker.patch("src.train_stock_model.load_config", return_value={"some_config_key": "some_value"})
#     mocker.patch("src.train_stock_model.get_config_path", return_value=("test_data_path.csv", "test_model_path.pkl"))
#
#     # Mock version 2 dataset
#     data_v2 = pd.DataFrame({
#         "Open": [1, 2, 3, 4, 5],
#         "High": [2, 3, 4, 5, 6],
#         "Low": [0, 1, 2, 3, 4],
#         "Volume": [100, 200, 300, 400, 500],
#         "Close_pct_change": [1.1, 2.1, 3.1, 4.1, 5.1],
#         "Close_ma_3": [None, None, 2.0, 3.0, 4.0],
#         "Close_ma_7": [None, None, None, None, 2.8],
#         "Close_lag_1": [None, 1.0, 2.0, 3.0, 4.0],
#     })
#     mocker.patch("src.train_stock_model.load_data", return_value=data_v2)
#
#     # Mock MLflow run
#     mock_start_run = mocker.patch("mlflow.start_run", return_value=MagicMock())
#
#     # Run the function
#     train_and_log_runs("mock_config.yaml")
#
#     # Validate that MLflow runs were started
#     assert mock_start_run.call_count == 4  # 3 runs + 1 nested for the best model
