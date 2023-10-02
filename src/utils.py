import pickle
from typing import Any


def serialize_object(obj: Any, filename: str) -> None:
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)


def load_pickle(filename: str) -> Any:
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
