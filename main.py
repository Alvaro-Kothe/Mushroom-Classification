import mlflow
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel

from src.env import ENCODER_PATH, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.utils import load_pickle


def get_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{EXPERIMENT_NAME}/latest"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model


def get_encoder():
    return load_pickle(ENCODER_PATH)


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
