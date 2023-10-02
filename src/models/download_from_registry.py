import argparse
from typing import Optional, Sequence

import mlflow

from src.env import LOCAL_MODEL_PATH, MLFLOW_TRACKING_URI
from src.models.utils import get_model_from_registry
from src.utils import serialize_object


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = get_model_from_registry()

    out_loc = args.output or LOCAL_MODEL_PATH

    serialize_object(model, out_loc)


if __name__ == "__main__":
    main()
