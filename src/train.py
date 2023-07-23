import argparse

import mlflow
from prefect import flow
from prefect.task_runners import SequentialTaskRunner

from src import data, models
from src.env import (
    ENCODER_PATH,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    NUM_TRIALS,
    TOP_N,
)
from src.utils import serialize_object


@flow(task_runner=SequentialTaskRunner())
def train_flow(data_path):
    df = data.read_data(data_path)
    (features, target), enc = data.prepare_data(df)

    if not ENCODER_PATH:
        raise ValueError("ENCODER_PATH environment variable is not defined.")
    serialize_object(enc, ENCODER_PATH)

    train, valid, test = data.split_data(features, target)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models.optimize_logistic(*train, *valid, NUM_TRIALS)
    models.optimize_xgboost(*train, *valid, NUM_TRIALS)
    models.register_best_model(*test, TOP_N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", required=True)
    args = parser.parse_args()

    train_flow(args.input_path)


if __name__ == "__main__":
    main()
