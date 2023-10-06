import pathlib
import os
import wine_prediction

PACKAGE_ROOT = pathlib.Path(wine_prediction.__file__).resolve().parent

DATASET_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

TARGET = "quality"

SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")

MODEL_NAME = "wine_model.pkl"

ALPHA = 0.7
L1_RATIO = 0.4

TEST_SIZE = 0.3

RANDOM_SEED = 6

