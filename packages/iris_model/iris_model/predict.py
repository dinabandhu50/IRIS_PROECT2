import numpy as np
import pandas as pd

from iris_model.preprocessing.data_management import load_pipeline
from iris_model.config import config

pipeline_file_name = "iris_model.pkl"
_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:

    data = pd.read_json(input_data)
    data.columns = ['x1', 'x2', 'x3', 'x4', 'y']

    prediction = _pipe.predict(data[config.FEATURES])
    response = {'predictions': prediction}

    return response
