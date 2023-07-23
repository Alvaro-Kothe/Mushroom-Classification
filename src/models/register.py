import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from prefect import task
from sklearn.metrics import accuracy_score

from src.env import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, TOP_N
from src.utils import load_pickle


def log_acc_test(X_test, y_test, run_id):
    """Load model and log test accuracy in the run."""
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    test_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_accuracy", test_accuracy)


@task
def register_best_model(X_test, y_test, top_n: int):
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.log_loss ASC"],
    )
    for run in runs:
        log_acc_test(X_test, y_test, run.info.run_id)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id, order_by=["metrics.test_accuracy DESC"]
    )[0]

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=EXPERIMENT_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-data", required=True)
    parser.add_argument("-n", "--top-n", type=int)
    args = parser.parse_args()

    X_test, y_test = load_pickle(args.input_data)
    top_n = args.top_n or TOP_N

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    register_best_model.fn(X_test, y_test, top_n)


if __name__ == "__main__":
    main()
