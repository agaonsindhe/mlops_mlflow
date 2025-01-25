"""
This module contains mlflow utility functions for the stock model project.
"""
import mlflow
import mlflow.sklearn

def log_params_and_metrics(params, metrics):
    """
    Log parameters and metrics to MLflow.
    """
    for param, value in params.items():
        mlflow.log_param(param, value)

    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

def log_model(model, model_name):
    """
    Log a model artifact to MLflow.
    """
    mlflow.sklearn.log_model(model, model_name)
