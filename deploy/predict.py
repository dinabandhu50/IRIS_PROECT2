import pandas as pd

import joblib
import config


def make_prediction(input_data):
    _pipe = joblib.load(filename=config.PIPELINE_NAME)

    results = _pipe.predict(input_data)

    return results


if __name__ == '__main__':

    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # data
    data = pd.read_csv(config.TRAINING_DATA_FILE)
    data.columns = ['x1', 'x2', 'x3', 'x4', 'y']

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=1
    )

    pred = make_prediction(X_test)

    # # output transform
    # _le = joblib.load(filename=config.OUTPUT_TRANSFORM_NAME)

    # # # pred transform
    # _pred = _le.transform(pred)
    # print(y_test)
    # print(_pred)
    # determine matrics
    print(f'test accuracy {accuracy_score(y_test, pred)}')
    print(classification_report(y_test, pred))
