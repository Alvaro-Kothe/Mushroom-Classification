from src.mushroom import MUSHROOM_CHARACTERISTICS, Mushroom


def test_mushroom_fields():
    assert MUSHROOM_CHARACTERISTICS.keys() == Mushroom.__fields__.keys()
