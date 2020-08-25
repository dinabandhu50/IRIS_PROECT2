import pathlib
import iris_model

# iris_model.__file__ : This is the path of iris_model
# resolve is to print out the path
# parent gives the path of resolved path
PACKAGE_ROOT = pathlib.Path(iris_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TRAINING_DATA_FILE = 'iris.csv'

TARGET = 'y'

# Feature engineering
FEATURES = ['x1', 'x2', 'x3', 'x4']

NUMERIC_COL = ['x1', 'x2', 'x3', 'x4']

# Model saving
PIPELINE_NAME = './models/model_iris'
OUTPUT_TRANSFORM_NAME = './models/label_enc'
