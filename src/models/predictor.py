import json
import pickle as pkl
import os
from pprint import pprint

import mlflow.pyfunc
import numpy as np
import tensorflow as tf
from fastapi import HTTPException
from dotenv import load_dotenv
from mlflow import MlflowClient

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
        loaded_models = client.search_registered_models()
        for rm in loaded_models:
            self.model_map[rm.name] = mlflow.pyfunc.load_model(model_uri=f'models:/{rm.name}/latest')

        self.target_scaler_map = {}


    def predict_station(self, station_name, data):
        reshaped_data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

        model = mlflow.pyfunc.load_model(model_uri=f'models:/{station_name}.csv/latest')
        pprint(dict(model.metadata))
        return 3

        prediction = self.model_map[f'{station_name}.csv'].predict(reshaped_data)[0]
        prediction = self.target_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        return __adapt_predictions__(prediction)

    def return_model_map(self):
        return self.model_map
