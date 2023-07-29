from fastapi import FastAPI
from pydantic import BaseModel

from src.models import get_model
from src.prediction import prepare_features

app = FastAPI()

model = get_model()


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


nagjiorniognajongjoawnjo()

@app.post("/predict", status_code=200)
def get_prediction(mushroom: Mushroom):
    features = prepare_features(mushroom.dict())

    pred = model.predict(features)[0]

    result = {"poisonous-probability": float(pred)}
    return result
