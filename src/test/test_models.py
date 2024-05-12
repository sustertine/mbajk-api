import pytest
from src.models.input_model import InputModel


def test_input_model():
    input_model = InputModel()

    # Assert that the default values are as expected
    assert input_model.temperature == 25.1
    assert input_model.relative_humidity == 45
    assert input_model.dew_point == 12.4
    assert input_model.apparent_temperature == 24.7
    assert input_model.precipitation_probability == 0.0
    assert input_model.rain == 0.0
    assert input_model.surface_pressure == 984.3
    assert input_model.bike_stands == 22
    assert input_model.available_bike_stands == 8


def test_input_transformer():
   pass


def test_predictor():
    pass
