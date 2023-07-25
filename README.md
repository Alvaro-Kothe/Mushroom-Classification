# MLOps Zoomcamp Project — Mushroom Classification

## Dataset

[Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification)


## Tools used

- Poetry
- Make
- Pyenv
- Prefect
- MLFlow
- dotenv
- pre-commit
- FastAPI
- AWS
- Docker

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
$ docker compose up
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
$ curl -X 'POST' \
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

The response object is a json object with the probability of the mushroom be poisonous,
the response for the object above is

```bash
{"poisonous-probability":0.0}
```

## Build locally


Activate environment:
```bash
# if using poetry
$ poetry shell

# if using venv
$ source venv/bin/activate
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
$ prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
```
Start prefect server:
```bash
$ prefect server start
```

Start mlflow server in another window (also reactivate the python environment):
```bash
$ mlflow server --backend-store-uri sqlite:///mlflow.db
```

Train model:
```bash
$ python src/train.py -i data/mushrooms.csv
```

Start web-service:
```bash
$ uvicorn main:app --reload
```
