import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

import pipeline
import config


def run_training():
    # read data
    df = pd.read_csv(config.TRAINING_DATA_FILE)
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
    joblib.dump(pipeline.pipe, config.PIPELINE_NAME)
    joblib.dump(le, config.OUTPUT_TRANSFORM_NAME)


if __name__ == '__main__':
    run_training()
