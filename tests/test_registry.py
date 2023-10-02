from unittest.mock import Mock, patch

import pytest

from src.models import register


@pytest.fixture(name="mock_mlflow")
def setup_mlflow():
    with patch("src.models.register.mlflow") as mock_mlflow:
        mock_mlflow.pyfunc.load_model.return_value = Mock()
        mock_mlflow.MlflowClient().search_runs.return_value = [Mock(), Mock(), Mock()]
        yield


def test_register_best_model_run(mock_mlflow):
    # pylint: disable=unused-argument
    with patch.object(register, "log_acc_test"):
        register.register_best_model.fn(..., ..., 5)


def test_log_acc_run(mock_mlflow):
    # pylint: disable=unused-argument
    with patch("src.models.register.accuracy_score", return_value=0.5):
        register.log_acc_test(..., ..., 5)


@patch("src.models.register.load_pickle")
def test_main_register(mock_pickle, mock_mlflow):
    # pylint: disable=unused-argument
    mock_pickle.return_value = None, None

    with patch("src.models.register.register_best_model", Mock()) as mock_reg:
        register.main(["--input-data", "foo.pkl"])
        mock_reg.fn.assert_called_once()
