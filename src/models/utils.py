from typing import Any

import mlflow

from src.env import (
    ENCODER_PATH,
    EXPERIMENT_NAME,
    LOCAL_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    USE_MLFLOW,
)
from src.utils import load_pickle


def get_model_from_registry() -> Any:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{EXPERIMENT_NAME}/latest"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model


def get_model_local() -> Any:
    return load_pickle(LOCAL_MODEL_PATH)


def get_model() -> Any:
    if USE_MLFLOW:
        return get_model_from_registry()
    return get_model_local()


def get_encoder() -> Any:
    return load_pickle(ENCODER_PATH)
