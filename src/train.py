import argparse

import mlflow
from prefect import flow
from prefect.task_runners import SequentialTaskRunner

from src import data, models


@flow(task_runner=SequentialTaskRunner())
def train_flow(data_path):
    df = data.read_data(data_path)
    (features, target), _ = data.prepare_data(df)
    train, valid, test = data.split_data(features, target)
    mlflow.set_tracking_uri(models.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(models.EXPERIMENT_NAME)

    models.optimize_logistic(*train, *valid, models.NUM_TRIALS)
    models.optimize_xgboost(*train, *valid, models.NUM_TRIALS)
    models.register_best_model(*test, models.TOP_N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", required=True)
    args = parser.parse_args()

    train_flow(args.input_path)


if __name__ == "__main__":
    main()
