"""Utilities for MLflow."""

import ast
from pathlib import Path

from mlflow.tracking import MlflowClient


def read_mlflow_params(
    trained_model_path: str, tracking_uri: str = None
) -> dict:
    """Read parameters for a specific MLflow run."""
    # Create MLflow client
    mlruns_path = str(Path(trained_model_path).parents[3])
    client = MlflowClient(tracking_uri=mlruns_path)

    # Get the run
    runID = Path(trained_model_path).parents[1].stem
    run = client.get_run(runID)

    # Access parameters
    params = run.data.params
    params["run_name"] = run.info.run_name

    return params


def read_config_from_mlflow_params(mlflow_params: dict) -> dict:
    """Read config from MLflow parameters."""
    config = {
        k.removeprefix("config/"): ast.literal_eval(v)
        for k, v in mlflow_params.items()
        if k.startswith("config/")
    }
    return config


def read_cli_args_from_mlflow_params(mlflow_params: dict) -> dict:
    """Read CLI arguments from MLflow parameters."""
    cli_args = {
        k.removeprefix("cli_args/"): safe_eval_string(v)
        for k, v in mlflow_params.items()
        if k.startswith("cli_args/")
    }
    return cli_args


def safe_eval_string(s):
    """Try to evaluate a string as a literal, otherwise return as-is."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # return as-is if not a valid literal
        return s
