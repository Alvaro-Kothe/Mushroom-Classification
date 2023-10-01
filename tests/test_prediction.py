# pylint: disable=duplicate-code

import numpy as np
import pytest

from src.prediction.prepare_features import prepare_features


def test_prepare_features():
    correct_dict = {
        "cap_shape": "x",
        "cap_surface": "s",
        "cap_color": "n",
        "bruises": "t",
        "odor": "a",
        "gill_attachment": "f",
        "gill_spacing": "c",
        "gill_size": "n",
        "gill_color": "b",
        "stalk_shape": "e",
        "stalk_root": "e",
        "stalk_surface_above_ring": "f",
        "stalk_surface_below_ring": "f",
        "stalk_color_above_ring": "b",
        "stalk_color_below_ring": "b",
        "veil_type": "p",
        "veil_color": "n",
        "ring_number": "n",
        "ring_type": "e",
        "spore_print_color": "k",
        "population": "a",
        "habitat": "g",
    }

    wrong_dict = {
        "cap_shape": "p",
        "cap_surface": "p",
        "cap_color": "p",
        "bruises": "p",
        "odor": "p",
        "gill_attachment": "p",
        "gill_spacing": "p",
        "gill_size": "p",
        "gill_color": "p",
        "stalk_shape": "p",
        "stalk_root": "p",
        "stalk_surface_above_ring": "p",
        "stalk_surface_below_ring": "p",
        "stalk_color_above_ring": "p",
        "stalk_color_below_ring": "p",
        "veil_type": "p",
        "veil_color": "p",
        "ring_number": "p",
        "ring_type": "p",
        "spore_print_color": "p",
        "population": "p",
        "habitat": "p",
    }

    correct_features = prepare_features(correct_dict)

    assert isinstance(correct_features, np.ndarray)

    num_cat = np.array(
        [6, 4, 10, 2, 9, 2, 2, 2, 12, 2, 5, 4, 4, 9, 9, 1, 4, 3, 5, 9, 6, 7]
    )
    assert np.all(0 <= correct_features) and np.all(correct_features < num_cat)

    with pytest.raises(ValueError):
        prepare_features(wrong_dict)
