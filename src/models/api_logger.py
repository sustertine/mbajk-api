import os
from datetime import datetime
from typing import List

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

class ApiLogEntry():
    def __init__(self):
        self.station_name = None
        self.available_bike_stands = None
        self.date = None
        self.temperature = None
        self.relative_humidity = None
        self.dew_point = None
        self.apparent_temperature = None
        self.precipitation_probability = None
        self.rain = None
        self.surface_pressure = None
        self.predictions = []

    def to_df(self):
        return pd.DataFrame([self.dict()])

    @staticmethod
    def from_df(df):
        entry = ApiLogEntry()
        entry.available_bike_stands = int(df['available_bike_stands'].values[0])
        entry.date = datetime.now().isoformat()
        entry.temperature = float(df['temperature'].values[0])
        entry.relative_humidity = int(df['relative_humidity'].values[0])
        entry.dew_point = float(df['dew_point'].values[0])
        entry.apparent_temperature = float(df['apparent_temperature'].values[0])
        entry.precipitation_probability = int(df['precipitation_probability'].values[0])
        entry.rain = int(df['rain'].values[0])
        entry.surface_pressure = float(df['surface_pressure'].values[0])

        return entry

    def to_dict(self):
        return {
            'station_name': self.station_name,
            'available_bike_stands': self.available_bike_stands,
            'date': self.date,
            'temperature': self.temperature,
            'relative_humidity': self.relative_humidity,
            'dew_point': self.dew_point,
            'apparent_temperature': self.apparent_temperature,
            'precipitation_probability': self.precipitation_probability,
            'rain': self.rain,
            'surface_pressure': self.surface_pressure,
            'predictions': self.predictions
        }

class ApiLogger():
    def __init__(self):
        self.mongo_uri = os.getenv('MONGO_URL')
        self.client = MongoClient(self.mongo_uri)

    def log(self, data: ApiLogEntry):
        db = self.client['mbajk-api-calls']
        collection = db['mbajk-api-calls']
        collection.insert_one(data.to_dict())



