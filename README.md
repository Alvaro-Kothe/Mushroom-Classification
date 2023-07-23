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

## Run locally

Set prefect api to local:
```bash
$ poetry run prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
```
Start prefect server:
```bash
$ poetry run prefect server start
```

Start mlflow server in another window:
```bash
$ poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --artifacts-destination artifacts/
```

Train model:
```bash
$ python src/train.py -i data/mushrooms.csv
```
