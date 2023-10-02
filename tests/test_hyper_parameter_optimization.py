from unittest.mock import patch

import numpy as np
import pytest

from src.models import hpo


@pytest.fixture(name="mock_mlflow", autouse=True, scope="session")
def setup_mocked_mlflow():
    with patch.object(hpo, "mlflow"):
        yield


@pytest.fixture(name="data")
def generate_data():
    rng = np.random.default_rng(seed=42)
    features = rng.normal(size=(10, 4))
    target = rng.binomial(n=1, p=0.5, size=10)

    return features, target, features, target


def test_optimize_logistic(data):
    X_train, y_train, X_val, y_val = data
    hpo.optimize_logistic.fn(X_train, y_train, X_val, y_val, 2)


def test_optimize_xgboost(data):
    X_train, y_train, X_val, y_val = data
    hpo.optimize_xgboost.fn(X_train, y_train, X_val, y_val, 2)


def test_hpo_main_no_val():
    with (
        patch.object(hpo, "optimize_logistic") as mock_log,
        patch.object(hpo, "optimize_xgboost") as mock_xgb,
        patch.object(hpo, "load_pickle") as mock_load,
    ):
        mock_load.return_value = None, None
        hpo.main(["--train-data", "foo.pkl", "--num-trials", "50"])

        mock_log.fn.assert_called_with(None, None, None, None, 50)
        mock_xgb.fn.assert_called_with(None, None, None, None, 50)


def test_hpo_main_with_val():
    with (
        patch.object(hpo, "optimize_logistic") as mock_log,
        patch.object(hpo, "optimize_xgboost") as mock_xgb,
        patch.object(hpo, "load_pickle") as mock_load,
    ):
        mock_load.return_value = None, None
        hpo.main(
            ["--train-data", "foo.pkl", "--val-data", "bar.pkl", "--num-trials", "50"]
        )

        mock_log.fn.assert_called_with(None, None, None, None, 50)
        mock_xgb.fn.assert_called_with(None, None, None, None, 50)
