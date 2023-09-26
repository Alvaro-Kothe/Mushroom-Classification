# MLOps Zoomcamp Project — Mushroom Classification

My final project for [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)

## Objective

The goal of this project is to develop and build a MLOps pipeline
to build and deploy a predictive model to determine the edibility of mushrooms based on their characteristics.

## Dataset

The [dataset](data/mushrooms.csv)
used in this project has been downloaded from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).
This dataset includes descriptions of hypothetical samples corresponding to 23
 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from
  The Audubon Society Field Guide to North American Mushrooms (1981).
Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one.

## Tools used

- Poetry — Python depedency manager
- Pyenv — Python version manager
- Prefect — Workflow orchestrator
- MLFlow — Experiment tracker and model register
- FastAPI — Web API
- dotenv — environment variable loader
- pre-commit — pre-commit hooks
- AWS — Cloud service
- Docker — Containerization

## Pre-requisites

### Credentials

To change the default behaviour or use a cloud server,
copy `.env.example` to `.env` with

```bash
cp .env.example .env
```

And change the default values to your needs.

## Build Docker Image

It is possible to build the image with `docker compose` or `docker build`

### Docker Compose

To build and run the image run

```bash
docker compose up
```

### Docker Build

To build the Docker Image run

```bash
make build
```

To launch the application run

```bash
docker run -it --rm -p 8000:8000 mushroom-classification
```

## Using the API

The application works on POST requests, to send a request with CURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
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
'
```

The features and its possible values to be used in the API
can be seen in [docs/data.md](docs/data.md).

The response object is a json object with the probability of the mushroom be poisonous,
the response for the object above is

```bash
{"poisonous-probability":0.0}
```

## Build locally

Activate environment:

```bash
# if using poetry
poetry shell

# if using venv
source venv/bin/activate
```

- Install with poetry:

    ```bash
    poetry install
    ```

- Install with pip

    Activate the environment
    and run:

    ```bash
    pip install .
    ```

Set prefect api to local:

```bash
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
```

Start prefect server:

```bash
prefect server start
```

Start mlflow server in another window (also reactivate the python environment):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db
```

Train model:

```bash
python src/train.py -i data/mushrooms.csv
```

Start web-service:

```bash
uvicorn src.api:app --reload
```

## Further improvements

- [ ] Add a monitoring service
- [ ] Create a Frontend for the API
- [ ] Implement IaC
- [x] Use CI/CD
- [x] Create tests

## Disclaimer

The prediction model was created solely with the purpose in create a MLOps pipeline
and is not advisable to use the deployed model with unknown mushrooms.
