from typing import Any

from src.models.utils import get_encoder


def prepare_features(features: dict[str, str]) -> Any:
    enc = get_encoder()
    return enc.transform([list(features.values())])
