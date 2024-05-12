import os

import mlflow.pyfunc
import numpy as np
from fastapi import HTTPException
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn import set_config

set_config(transform_output="pandas")

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')

client = MlflowClient()


def __adapt_predictions__(predictions):
    predictions = np.round(predictions)
    predictions[predictions < 0] = 0
    return predictions


class Predictor:
    def __init__(self):

        self.model_map = {}
        self.targt_scaler_map = {}
        self.pipeline_map = {}

        loaded_models = client.search_registered_models()

        for rm in loaded_models:
            model_uri = client.get_model_version_download_uri(rm.name, rm.latest_versions[0].version)
            model_info = mlflow.models.Model.load(model_uri)
            model_flavors = model_info.flavors.keys()

            if 'python_function' in model_flavors:
                self.model_map[rm.name] = mlflow.pyfunc.load_model(model_uri=f'models:/{rm.name}/latest')
            elif 'sklearn' in model_flavors:
                self.model_map[rm.name] = mlflow.sklearn.load_model(model_uri=f'models:/{rm.name}/latest')


    def predict_station(self, station_name, data):
        if f'{station_name}.csv' not in self.model_map:
            raise HTTPException(status_code=404, detail=f"Model for station {station_name} not found")

        target = data['available_bike_stands']

        data = self.model_map[f'{station_name}.csv_pipeline'].transform(data)
        data['available_bike_stands'] = target

        reshaped_data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

        prediction = self.model_map[f'{station_name}.csv'].predict(reshaped_data)[0]
        prediction = self.model_map[f'{station_name}.csv_target_scaler'].inverse_transform(
            prediction.reshape(-1, 1)).flatten()
        return __adapt_predictions__(prediction)
