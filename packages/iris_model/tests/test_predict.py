import math

from iris_model.predict import make_prediction
from iris_model.preprocessing.data_management import load_dataset
from iris_model.config import config


def test_print():
    print('Inside test predict')


def test_make_single_prediction():
    #     # Given
    test_data = load_dataset(
        file_name=config.TRAINING_DATA_FILE)
    single_test_json = test_data[10:11].to_json(orient='records')

    # when
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], str)
    assert subject.get('predictions')[0] == 'Iris-setosa'
