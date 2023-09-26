from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.models import get_model
from src.mushroom import MUSHROOM_CHARACTERISTICS, Mushroom
from src.prediction import prepare_features

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = get_model()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    context = {"request": request, "characteristics": MUSHROOM_CHARACTERISTICS}
    return templates.TemplateResponse("index.html", context)


@app.post("/predict", status_code=200)
def get_prediction(mushroom: Mushroom):
    features = prepare_features(mushroom.dict())

    pred = model.predict(features)[0]

    result = {"poisonous-probability": float(pred)}
    return result
