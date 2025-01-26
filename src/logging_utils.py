"""
this is logging utils file
"""
import matplotlib.pyplot as plt
import mlflow


def log_predicted_vs_actual(y_test, y_pred, run_name="Predicted vs Actual"):
    """
    Logs a scatter plot of predicted vs actual values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Fit")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(run_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png")
    plt.close()

    # Log the graph in MLflow
    mlflow.log_artifact("predicted_vs_actual.png", artifact_path="plots")


def log_residual_plot(y_test, y_pred, run_name="Residual Plot"):
    """
    Logs a residual plot for the model predictions.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, label="Residuals")
    plt.axhline(0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(run_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("residual_plot.png")
    plt.close()

    # Log the graph in MLflow
    mlflow.log_artifact("residual_plot.png", artifact_path="plots")


def log_metric_trend(runs, metric_name="rmse", run_name="Metric Trend"):
    """
    Logs a trend graph for a specific metric over multiple runs.
    """
    metrics = [run.get(metric_name) for run in runs if metric_name in run]
    if not metrics:
        raise ValueError(f"No runs contain the metric '{metric_name}'.")
    print(f"Metric trend for {metric_name}: {metrics}")
    run_ids = [i + 1 for i in range(len(runs))]

    plt.figure(figsize=(8, 6))
    plt.plot(run_ids, metrics, marker="o", label=f"{metric_name.upper()} Trend")
    plt.xlabel("Run Index")
    plt.ylabel(metric_name.upper())
    plt.title(run_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{metric_name}_trend.png")
    plt.close()

    # Log the graph in MLflow
    mlflow.log_artifact(f"{metric_name}_trend.png", artifact_path="plots")
