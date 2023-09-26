from src.mushroom import MUSHROOM_CHARACTERISTICS, Mushroom


def test_mushroom_fields():
    assert MUSHROOM_CHARACTERISTICS.keys() == Mushroom.__fields__.keys()


def test_valid_fields_for_encoding():
    unique_values = {
        "class": ["p", "e"],
        "cap_shape": ["x", "b", "s", "f", "k", "c"],
        "cap_surface": ["s", "y", "f", "g"],
        "cap_color": ["n", "y", "w", "g", "e", "p", "b", "u", "c", "r"],
        "bruises": ["t", "f"],
        "odor": ["p", "a", "l", "n", "f", "c", "y", "s", "m"],
        "gill_attachment": ["f", "a"],
        "gill_spacing": ["c", "w"],
        "gill_size": ["n", "b"],
        "gill_color": ["k", "n", "g", "p", "w", "h", "u", "e", "b", "r", "y", "o"],
        "stalk_shape": ["e", "t"],
        "stalk_root": ["e", "c", "b", "r", "?"],
        "stalk_surface_above_ring": ["s", "f", "k", "y"],
        "stalk_surface_below_ring": ["s", "f", "y", "k"],
        "stalk_color_above_ring": ["w", "g", "p", "n", "b", "e", "o", "c", "y"],
        "stalk_color_below_ring": ["w", "p", "g", "b", "n", "e", "y", "o", "c"],
        "veil_type": ["p"],
        "veil_color": ["w", "n", "o", "y"],
        "ring_number": ["o", "t", "n"],
        "ring_type": ["p", "e", "l", "f", "n"],
        "spore_print_color": ["k", "n", "u", "h", "w", "r", "o", "y", "b"],
        "population": ["s", "n", "a", "v", "y", "c"],
        "habitat": ["u", "g", "m", "d", "p", "w", "l"],
    }

    for field, codes in MUSHROOM_CHARACTERISTICS.items():
        uniques = unique_values[field]

        assert set(codes.values()) == set(uniques), f"values not found in {field}"
