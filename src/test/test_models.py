import pytest
import pandas as pd
from src.models.input_model import InputModel
from src.models.predictor import Predictor
from src.models.input_model import InputModel
from src.models.input_transformer import InputTransformer


def test_input_model():
    # Initialize an instance of InputModel
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
    # Instantiate the InputTransformer
    transformer = InputTransformer()

    # Create a sample InputModel instance
    data = InputModel(
        temperature=20.0,
        dew_point=10.0,
        apparent_temperature=18.0,
        surface_pressure=1000.0,
        available_bike_stands=5,
        relative_humidity=0.5,
        precipitation_probability=0.1,
        rain=0.0
    )

    # Transform the data
    transformed_data = transformer.transform(data)

    # Check if the output is a pandas DataFrame
    assert isinstance(transformed_data, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = ['temperature', 'dew_point', 'apparent_temperature',
                        'surface_pressure', 'available_bike_stands',
                        'lagged_available_bike_stands', 'rolling_mean_bike_stands',
                        'rolling_std_bike_stands', 'diff_available_bike_stands',
                        'temperature_diff']
    assert set(transformed_data.columns) == set(expected_columns)


def test_predictor():
    # Initialize an instance of Predictor
    predictor = Predictor()

    # Assert that the model and target_scaler are not None
    assert predictor.model is not None
    assert predictor.target_scaler is not None
