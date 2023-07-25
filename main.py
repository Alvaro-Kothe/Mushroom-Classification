from fastapi import FastAPI
from pydantic import BaseModel

from src.models import get_encoder, get_model


def prepare_features(features: dict):
    # replace underscore with hyphen
    # new_dict = {
    #         str(k).replace('_', '-'): v for
    #         k, v in features.items()
    #         }
    enc = get_encoder()
    # print(new_dict)
    return enc.transform([list(features.values())])


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Mushroom(BaseModel):
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str


@app.post("/predict", status_code=200)
def get_prediction(mushroom: Mushroom):
    features = prepare_features(mushroom.dict())

    model = get_model()

    pred = model.predict(features)[0]

    result = {"poisonous-probability": float(pred)}
    return result