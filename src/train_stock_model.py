"""
This module contains functions to train and evaluate a Linear Regression model
for stock price prediction using historical data.
"""
import pickle
import dvc.api
from datetime import datetime
from math import sqrt
import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from src.dataset_utils import load_data, get_dataset_version
from src.preprocess_data import preprocess_data
from src.utils import load_config, get_config_path
from src.logging_utils import log_predicted_vs_actual, log_residual_plot, log_metric_trend

global input_example, signature, x_test, y_test
def evaluate_model(model, x_test_param, y_test_param):
    """
    Evaluate the model on test data and return metrics.
    """
    y_pred = model.predict(x_test_param)
    mse = mean_squared_error(y_test_param, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test_param, y_pred)
    evs = explained_variance_score(y_test_param, y_pred)
    r2 = r2_score(y_test_param, y_pred)

    return {"rmse": rmse, "mae": mae, "evs": evs, "r2": r2}

def train_and_log_runs(config_path="config.yaml"):
    """
    Train and log three different Ridge regression models with varying hyperparameters and log the best model.
    """
    # Start MLflow Experiment

    experiment_name = f"Stock Price Prediction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    mlflow.set_experiment(experiment_name)


    # Load configuration
    config = load_config(config_path)

    # Get the dataset path dynamically
    data_path, model_path = get_config_path(config)

    dataset_version = get_dataset_version(data_path)

    # Load and preprocess data
    df = load_data(data_path)

    df = preprocess_data(df)

    # Dynamically select features based on existing columns
    required_features = ['Open', 'High', 'Low', 'Volume', 'Close_ma_3', 'Close_ma_7', 'Close_lag_1']
    features = [col for col in required_features if col in df.columns]
    random_states = [42, 84, 123]
    if not features:
        raise ValueError("No valid features found in the dataset. Check the dataset for missing columns.")

    target = 'Close_pct_change'
    if 'Close_pct_change' not in df.columns:
        df['Close_pct_change'] = df['Close'].pct_change()

    # Drop NaN values and align indices between features and target
    features = df[features].dropna()
    target = df[target].dropna()

    # Align indices to ensure consistent rows
    common_indices = features.index.intersection(target.index)
    features = features.loc[common_indices]
    target = target.loc[common_indices]

    # Define hyperparameters for three runs
    runs = [
        {"alpha": 0.1, "solver": "auto"},
        {"alpha": 1.0, "solver": "svd"},
        {"alpha": 10.0, "solver": "cholesky"},
    ]

    best_model = None
    best_metrics = None
    best_run_id = None

    # Run experiments
    for i, params in enumerate(runs):

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=random_states[i])
        signature, input_example = infer_mlflow_signature(x_train)

        with mlflow.start_run(nested=True):
            # Initialize model with current parameters
            model = Ridge(alpha=params["alpha"], solver=params["solver"])
            model.fit(x_train, y_train)

            # Evaluate model
            metrics = evaluate_model(model, x_test, y_test)

            # Log parameters and metrics
            mlflow.log_param("run_index", i + 1)
            mlflow.log_param("alpha", params["alpha"])
            mlflow.log_param("solver", params["solver"])
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("evs", metrics["evs"])
            mlflow.log_metric("r2", metrics["r2"])

            # Log the trained model
            mlflow.sklearn.log_model(model, f"model_run_{i + 1}",input_example=input_example,signature=signature)
            # Append metrics to the runs object
            params.update(metrics)
            # Log graphs
            log_predicted_vs_actual(y_test, model.predict(x_test))
            log_residual_plot(y_test, model.predict(x_test))

            # Update the best model if applicable
            if best_metrics is None or best_metrics["rmse"] > metrics["rmse"]:
                best_model = model
                best_metrics = metrics
                best_run_id = mlflow.active_run().info.run_id

    # Log the best model as a separate artifact
    if best_model:
        with mlflow.start_run(run_id=best_run_id, nested=True):
            mlflow.log_param("best_model", True)
            mlflow.log_metric("best_rmse", best_metrics["rmse"])
            mlflow.log_metric("best_mae", best_metrics["mae"])
            mlflow.log_metric("best_evs", best_metrics["evs"])
            mlflow.log_metric("best_r2", best_metrics["r2"])
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)

            mlflow.sklearn.log_model(best_model, "best_model",input_example=input_example,signature=signature)
            mlflow.log_artifact("model.pkl",artifact_path="models")
            y_pred_best = best_model.predict(x_test)
            log_predicted_vs_actual(y_test, y_pred_best, run_name="Predicted vs Actual - Best Model")
            log_residual_plot(y_test, y_pred_best, run_name="Residual Plot - Best Model")

    print(f"Best model saved to {model_path}")
    print(f"Best metrics: {best_metrics}")


def infer_mlflow_signature(x_train):
    """
        Infers an MLflow signature for a given training dataset.

        Parameters:
        -----------
        x_train : array-like or DataFrame
            Training dataset where each row is an example and each column is a feature.
            The first row is used to create an input example.

        Returns:
        --------
        signature : mlflow.models.signature.ModelSignature
        input_example : array-like or DataFrame
    """
    input_example = x_train[:1]
    placeholder_model = Ridge(alpha=1.0)
    try:
        placeholder_predictions = placeholder_model.predict(input_example)
    except:
        placeholder_predictions = [0]
    signature = infer_signature(input_example, placeholder_predictions)

    return signature, input_example


if __name__ == "__main__":

    # Run training and evaluation
    train_and_log_runs("config.yaml")
