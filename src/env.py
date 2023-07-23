import os

from dotenv import load_dotenv

load_dotenv()

TOP_N = int(os.getenv("TOP_N") or 5)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "mushroom-classification")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
NUM_TRIALS = int(os.getenv("NUM_TRIALS") or 10)
ENCODER_PATH = os.getenv("ENCODER_PATH") or "./enc.pkl"
