from unittest.mock import patch

import pytest

from src.models import utils


class MockModel:
    pass


@pytest.fixture(name="mock_mlflow")
def setup_mocked_mlflow():
    with patch("src.models.utils.mlflow") as mock:
        mock.set_tracking_uri.return_value = None
        mock.pyfunc.load_model.return_value = MockModel()
        yield


def test_get_model_from_registry(mock_mlflow):
    # pylint: disable=unused-argument
    loaded_model = utils.get_model_from_registry()
    assert isinstance(loaded_model, MockModel)


def test_get_model():
    with patch("src.models.utils.USE_MLFLOW", True), patch(
        "src.models.utils.get_model_from_registry", return_value=MockModel()
    ):
        loaded_model = utils.get_model()
    assert isinstance(loaded_model, MockModel)
