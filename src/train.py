import argparse
from typing import Optional, Sequence

import mlflow
from prefect import flow
from prefect.task_runners import SequentialTaskRunner

from src.data.preprocess import prepare_data, read_data, split_data
from src.env import (
    ENCODER_PATH,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    NUM_TRIALS,
    TOP_N,
)
from src.models.hpo import optimize_logistic, optimize_xgboost
from src.models.register import register_best_model
from src.utils import serialize_object


@flow(task_runner=SequentialTaskRunner())
def train_flow(data_path: str) -> None:
    df = read_data(data_path)
    (features, target), enc = prepare_data(df)  # type: ignore

    if not ENCODER_PATH:
        raise ValueError("ENCODER_PATH environment variable is not defined.")
    serialize_object(enc, ENCODER_PATH)

    train, valid, test = split_data(features, target)  # type: ignore
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    optimize_logistic(*train, *valid, NUM_TRIALS)  # type: ignore
    optimize_xgboost(*train, *valid, NUM_TRIALS)  # type: ignore
    register_best_model(*test, TOP_N)  # type: ignore


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", required=True)
    args = parser.parse_args(argv)

    train_flow(args.input_path)


if __name__ == "__main__":
    main()
