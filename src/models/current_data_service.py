from io import StringIO

import dotenv
import os

import pandas as pd
import requests

dotenv.load_dotenv()


class CurrentDataService:
    def __init__(self):
        self.mbajk_url = os.getenv('MBAJK_URL')

    def get_current_data(self, station_name: str):
        mbajk_data = requests.get(self.mbajk_url)

        mbajk_df = pd.read_json(StringIO(mbajk_data.text))

        mbajk_df.drop(columns=['contract_name', 'number'], inplace=True, axis=1)
        mbajk_df['date'] = pd.to_datetime(mbajk_df['last_update'], unit='ms')

        mbajk_df = mbajk_df[mbajk_df['name'] == station_name]

        lat = mbajk_df['position'].iloc[0]['lat']
        lng = mbajk_df['position'].iloc[0]['lng']

        mbajk_df.drop(columns=['name', 'address', 'position', 'banking', 'bonus', 'bike_stands', 'available_bikes', 'status',
                         'last_update'], inplace=True)

        weather_df = self.get_weather(lat, lng)

        weather_df.rename(columns={
            'time': 'date',
            'temperature_2m': 'temperature',
            'relative_humidity_2m': 'relative_humidity',
            'dew_point_2m': 'dew_point'
        }, inplace=True)

        mbajk_df['date'] = pd.to_datetime(mbajk_df['date'])
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        merged_df = mbajk_df.apply(self.find_nearest, args=(weather_df,), axis=1)
        merged_df = pd.concat([mbajk_df, merged_df], axis=1)

        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        return merged_df

    def get_weather(self, lat, lng):
        response = requests.get(
            f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}1&hourly=temperature_2m,'
            f'relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure').json()
        return pd.DataFrame(response['hourly'])

    def get_available_bike_stands(self, station_name: str):
        current_data = self.get_current_data(station_name)
        return current_data['available_bike_stands'].iloc[0]

    def find_nearest(self, row, df, column='date'):
        exactmatch = df[df[column] == row[column]]
        if not exactmatch.empty:
            return exactmatch.iloc[0]
        else:
            lower = df[df[column] < row[column]].max()
            upper = df[df[column] > row[column]].min()
            nearest = lower if (row[column] - lower[column]) < (upper[column] - row[column]) else upper
            return nearest
