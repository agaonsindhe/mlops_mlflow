import pytest
import pandas as pd
from sklearn.linear_model import Ridge
from src.dataset_utils import load_data, add_features
from src.train_stock_model import evaluate_model
from src.utils import load_config, get_config_path


def test_load_data(config_path="config.yaml"):
    """
    Test that load_data loads the asset correctly.
    """

    # Load configuration
    config = load_config(config_path)

    # Get the dataset path dynamically
    data_path, model_path = get_config_path(config)

    # Load and preprocess data
    df = load_data(data_path)
    assert isinstance(df, pd.DataFrame), "load_data should return a DataFrame"


def test_add_features():
    """
    Test that add_features correctly adds new features.
    """

    # Load configuration
    config = load_config("config.yaml")

    # Get the dataset path dynamically
    data_path, model_path = get_config_path(config)

    # Load and preprocess data
    df = load_data(data_path)
    df_with_features = add_features(df.copy())

    # Check if features are added
    assert 'Close_ma_3' in df_with_features.columns, "Missing feature: Close_ma_3"
    assert 'Close_ma_7' in df_with_features.columns, "Missing feature: Close_ma_7"
    assert 'Close_lag_1' in df_with_features.columns, "Missing feature: Close_lag_1"
    assert 'Close_pct_change' in df_with_features.columns, "Missing feature: Close_pct_change"

    # Check if NaN rows are dropped
    assert len(df_with_features) < len(df), "Rows with NaN values should be dropped"


def test_evaluate_model():
    """
    Test the evaluate_model function.
    """

    # Load configuration
    config = load_config("config.yaml")

    # Get the dataset path dynamically
    data_path, model_path = get_config_path(config)

    # Load and preprocess data
    df = load_data(data_path)
    # Split the dataset into features and target
    x = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Fit a simple model
    model = Ridge()
    model.fit(x, y)

    # Evaluate the model
    metrics = evaluate_model(model, x, y)

    # Verify metrics are calculated correctly
    assert 'rmse' in metrics, "Missing metric: RMSE"
    assert 'r2' in metrics, "Missing metric: R²"
    assert 'mae' in metrics, "Missing metric: MAE"
    assert 'evs' in metrics, "Missing metric: EVS"
    assert metrics['rmse'] >= 0, "RMSE should be non-negative"
    assert 0 <= metrics['r2'] <= 1, "R² should be between 0 and 1"


def test_train_and_log_runs(mocker):
    """
    Test the train_and_log_runs function with mock MLflow logging.
    """
    from src.train_stock_model import train_and_log_runs

    # Mock MLflow functions
    mock_log_param = mocker.patch("mlflow.log_param")
    mock_log_metric = mocker.patch("mlflow.log_metric")
    mock_log_model = mocker.patch("mlflow.sklearn.log_model")


    # Run the training function
    train_and_log_runs("config.yaml")

    # Assert MLflow functions are called
    mock_log_param.assert_called()
    mock_log_metric.assert_called()
    mock_log_model.assert_called()
