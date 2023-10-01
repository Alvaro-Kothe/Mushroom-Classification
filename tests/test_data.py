# pylint: disable=redefined-outer-name
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder

from src.data.preprocess import main, prepare_data, read_data, split_data


@pytest.fixture
def data_path():
    return os.path.join(os.path.dirname(__file__), "data.csv")


@pytest.fixture
def data_frame():
    return pd.DataFrame({"class": ["p", "e"], "A": ["a", "b"], "B": ["c", "c"]})


def test_read_data_valid_file(data_path):
    df = read_data.fn(data_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_read_data_invalid_file():
    with pytest.raises(FileNotFoundError):
        read_data.fn("doesnotexist.csv")


def test_prepare_data(data_frame):
    (features, target), enc = prepare_data.fn(data_frame)
    assert isinstance(enc, OrdinalEncoder)
    assert len(features) == len(target) == 2
    np.testing.assert_array_equal(features, [[0.0, 0.0], [1.0, 0.0]])


def test_split():
    X = np.zeros((10, 10))
    y = np.ones(10)

    train, valid, test = split_data.fn(X, y, test_size=0.2)

    assert len(test[0]) == 2
    assert len(train[0]) == 6
    assert len(valid[0]) == 2


@patch("src.data.preprocess.read_data")
@patch("src.data.preprocess.prepare_data")
@patch("src.data.preprocess.split_data")
def test_main_preprocess(mock_split, mock_prepare, mock_read, tmp_path):
    mock_read.fn.return_value = None
    mock_prepare.fn.return_value = (None, None), None
    mock_split.fn.return_value = (None,) * 3
    main(["--input-path", "foo.csv", "--output-directory", os.fspath(tmp_path)])

    created_files = os.listdir(tmp_path)
    assert set(created_files) == set(["train.pkl", "valid.pkl", "test.pkl", "enc.pkl"])
