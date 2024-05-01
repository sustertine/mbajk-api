import pickle as pkl
import os

import pandas as pd

from src.models.input_model import InputModel


class InputTransformer:
    def __init__(self):
        models_dir = f'{os.getcwd()}/models'
        self.mms = pkl.load(open(f'{models_dir}/mms_scaler.pkl', 'rb'))
        self.std = pkl.load(open(f'{models_dir}/std_scaler.pkl', 'rb'))
        self.target_scaler = pkl.load(open(f'{models_dir}/target_scaler.pkl', 'rb'))

    def transform(self, data: InputModel):
        data = data.dict()
        data = {k: [v] for k, v in data.items()}

        df = pd.DataFrame(data)

        df['available_bike_stands'] = self.target_scaler.transform(
            df['available_bike_stands'].values.reshape(-1, 1))

        df[['temperature', 'dew_point', 'apparent_temperature', 'surface_pressure']] = self.std.transform(
            df[['temperature', 'dew_point', 'apparent_temperature', 'surface_pressure']])

        df[['relative_humidity', 'precipitation_probability', 'rain']] = self.mms.transform(
            df[['relative_humidity', 'precipitation_probability', 'rain']])

        df = df[['temperature', 'dew_point', 'apparent_temperature',
                 'surface_pressure', 'available_bike_stands']]

        # Set means since this is a single row
        df['lagged_available_bike_stands'] = 0.579272
        df['rolling_mean_bike_stands'] = 0.579318
        df['rolling_std_bike_stands'] = 0.049509
        df['diff_available_bike_stands'] = 0.000007
        df['temperature_diff'] = -5.458696e-16

        return df
