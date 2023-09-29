from unittest.mock import patch

from prefect.testing.utilities import prefect_test_harness

from src.train import train_flow


@patch("src.train.read_data")
@patch("src.train.prepare_data")
@patch("src.train.serialize_object")
@patch("src.train.split_data")
@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("src.train.optimize_logistic")
@patch("src.train.optimize_xgboost")
@patch("src.train.register_best_model")
@patch("src.train.TOP_N", 3)
@patch("src.train.NUM_TRIALS", 2)
@patch("src.train.EXPERIMENT_NAME", "test_experiment")
@patch("src.train.MLFLOW_TRACKING_URI", "baz")
def test_train_flow(
    mock_register,
    mock_xgboost,
    mock_logistic,
    mock_experiment,
    mock_tracking,
    mock_split,
    mock_serialize,
    mock_prepare,
    mock_read,
):
    mock_read.return_value = None
    mock_prepare.return_value = (None, None), None
    mock_split.return_value = (
        (None, None),
        (None, None),
        (None, None),
    )
    with prefect_test_harness():
        train_flow("foo.csv")

    mock_serialize.assert_called_once()
    mock_split.assert_called_once()
    mock_prepare.assert_called_once()
    mock_read.assert_called_once_with("foo.csv")
    mock_register.assert_called_once_with(None, None, 3)
    mock_experiment.assert_called_once_with("test_experiment")
    mock_logistic.assert_called_once_with(None, None, None, None, 2)
    mock_xgboost.assert_called_once_with(None, None, None, None, 2)
    mock_tracking.assert_called_once_with("baz")
