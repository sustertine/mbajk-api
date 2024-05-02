import pickle as pkl
import os

import numpy as np
import tensorflow as tf
from fastapi import HTTPException


def __adapt_predictions__(predictions):
    predictions = np.round(predictions)
    predictions[predictions < 0] = 0
    return predictions


class Predictor:
    def __init__(self):
        models_dir = f'{os.getcwd()}/models'
        self.model_map = {}
        self.target_scaler = pkl.load(open(f'{models_dir}/target_scaler.pkl', 'rb'))
        for filename in os.listdir(f'{models_dir}/stations'):
            station_name = filename.split('.')[0]
            self.model_map[station_name] = tf.keras.models.load_model(f'{models_dir}/stations/{filename}',
                                                                      compile=False)
        self.model = tf.keras.models.load_model(f'{models_dir}/basic_model.h5', compile=False)

    def predict(self, data):
        reshaped_data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        prediction = self.model.predict(reshaped_data)[0]
        prediction = self.target_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        return __adapt_predictions__(prediction)

    def predict_station(self, station_name, data):
        if station_name not in self.model_map:
            raise HTTPException(status_code=404, detail='Station not found')
        reshaped_data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        prediction = self.model_map[station_name].predict(reshaped_data)[0]
        prediction = self.target_scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        return __adapt_predictions__(prediction)
