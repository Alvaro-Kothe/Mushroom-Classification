from src.models.utils import get_encoder


def prepare_features(features: dict):
    enc = get_encoder()
    return enc.transform([list(features.values())])
