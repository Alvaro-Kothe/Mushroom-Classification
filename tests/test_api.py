from fastapi.testclient import TestClient

from src.api import app

def test_api():
    client = TestClient(app)

    request_data = {
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
            "habitat": "g"
            }

    response = client.post("/predict", json = request_data)
    assert response.status_code == 200

    response_json = response.json()
    assert 0 <= response_json["poisonous-probability" ] <= 1
