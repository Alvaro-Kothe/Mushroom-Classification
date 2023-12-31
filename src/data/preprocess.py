import argparse
import os.path
from typing import Any, Optional, Sequence, Tuple

import pandas as pd
from prefect import task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from src.typing import Tensor
from src.utils import serialize_object


@task
def read_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


@task
def prepare_data(
    df: pd.DataFrame,
) -> Tuple[Tuple[Tensor, Tensor], OrdinalEncoder]:
    """
    Separate Dataframe in features and target, where target is if mushroom is poisonous.
    """
    y = (df.pop("class") == "p").to_numpy()
    enc = OrdinalEncoder()
    X = enc.fit_transform(df.to_numpy())
    return (X, y), enc


@task
def split_data(
    X: Tensor, y: Tensor, test_size: float = 0.2
) -> tuple[tuple[Any, Any], tuple[Any, Any], tuple[Any, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=505
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=test_size, random_state=505
    )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Read Dataset and generates 4 files in `output_directory`:
    - `train.pkl`: Train data (features, target).
    - `valid.pkl`: Validation data (features, target).
    - `test.pkl`: Test data (features, target).
    - `enc.pkl`: OrdinalEncoder for features.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)

    df = read_data.fn(args.input_path)
    (features, target), enc = prepare_data.fn(df)
    train, valid, test = split_data.fn(features, target)
    serialize_object(train, os.path.join(args.output_directory, "train.pkl"))
    serialize_object(valid, os.path.join(args.output_directory, "valid.pkl"))
    serialize_object(test, os.path.join(args.output_directory, "test.pkl"))
    serialize_object(enc, os.path.join(args.output_directory, "enc.pkl"))


if __name__ == "__main__":
    main()
