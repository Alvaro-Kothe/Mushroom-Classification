from unittest.mock import patch

import pytest

from src.models import hpo


@pytest.fixture(name="mock_mlflow", autouse=True, scope="session")
def setup_mocked_mlflow():
    with patch.object(hpo, "mlflow"):
        yield


@pytest.fixture(name="mock_optuna")
def setup_mocked_optuna():
    with patch.object(hpo, "optuna"), patch.object(hpo, "TPESampler"):
        yield


def test_optimize_logistic(mock_optuna):
    # pylint: disable=unused-argument
    hpo.optimize_logistic.fn(..., ..., ..., ..., 5)


def test_optimize_xgboost(mock_optuna):
    # pylint: disable=unused-argument
    hpo.optimize_xgboost.fn(..., ..., ..., ..., 5)


def test_hpo_main():
    with (
        patch.object(hpo, "optimize_logistic") as mock_log,
        patch.object(hpo, "optimize_xgboost") as mock_xgb,
        patch.object(hpo, "load_pickle") as mock_load,
    ):
        mock_load.return_value = None, None
        hpo.main(["--train-data", "foo.pkl", "--num-trials", "50"])

        mock_log.fn.assert_called_with(None, None, None, None, 50)
        mock_xgb.fn.assert_called_with(None, None, None, None, 50)
