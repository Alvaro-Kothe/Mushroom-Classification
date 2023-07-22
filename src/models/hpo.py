import argparse
import os

import mlflow
import optuna
from dotenv import load_dotenv
from optuna.samplers import TPESampler
from prefect import task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from src.utils import load_pickle

load_dotenv()
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "mushroom-classification")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


@task
def optimize_logistic(X_train, y_train, X_val, y_val, num_trials):
    mlflow.sklearn.autolog()

    def objective(trial: optuna.Trial):
        with mlflow.start_run():
            params = {
                "C": trial.suggest_float("C", 1e-2, 1, log=True),
                "n_jobs": -1,
                "max_iter": 1000,
            }

            lr = LogisticRegression(**params)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_val)
            metric_log_loss = log_loss(y_val, y_pred)

            mlflow.log_metric("log_loss", metric_log_loss)

        return metric_log_loss

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


@task
def optimize_xgboost(X_train, y_train, X_val, y_val, num_trials):
    mlflow.xgboost.autolog()

    def objective(trial: optuna.Trial):
        with mlflow.start_run():
            params = {
                "objective": "binary:logistic",
                "max_depth": trial.suggest_int("max_depth", 1, 20),
                "min_child_weight": trial.suggest_int("min_child_weight", 0, 10),
                "gamma": trial.suggest_float("gamma", 1, 9),
            }
            clf = XGBClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            metric_log_loss = log_loss(y_val, y_pred)

            mlflow.log_metric("log_loss", metric_log_loss)

        return metric_log_loss

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data")
    parser.add_argument("-n", "--num-trials", type=int)
    args = parser.parse_args()

    X_train, y_train = load_pickle(args.train_data)

    if args.val_data is None:
        X_val, y_val = X_train, y_train
    else:
        X_val, y_val = load_pickle(args.val_data)

    num_trials = args.num_trials or os.getenv("NUM_TRIALS") or 10

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    optimize_logistic.fn(X_train, y_train, X_val, y_val, num_trials)
    optimize_xgboost.fn(X_train, y_train, X_val, y_val, num_trials)


if __name__ == "__main__":
    main()
