import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from iris_model import pipeline
from iris_model.config import config


def save_pipeline(*, pipeline_to_percist) -> None:

    save_file_name = "iris_model.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_percist, save_path)

    print("saved pipeline")


def run_training() -> None:
    # read data
    df = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE)
    df.columns = ['x1', 'x2', 'x3', 'x4', 'y']

    # train test split of data
    X_train, X_test, y_train, y_test = train_test_split(
        df[config.FEATURES],
        df[config.TARGET],
        test_size=0.1,
        random_state=1
    )

    # transform the target
    le = LabelEncoder()
    le.fit(y_train)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)

    # fitting the model
    pipeline.pipe.fit(X_train, y_train)
    # joblib.dump(pipeline.pipe, config.PIPELINE_NAME)
    # joblib.dump(le, config.OUTPUT_TRANSFORM_NAME)
    save_pipeline(pipeline_to_percist=pipeline.pipe)


if __name__ == '__main__':
    run_training()
