import pandas as pd
import requests
import os
from remove_headers import remove_header_rows


def fetch_weather(lat, lng):
    print(f'Fetching weather for {lat}, {lng}')
    url = (f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}1&hourly=temperature_2m,'
           f'relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure')
    response = requests.get(url).json()
    return response


def fetch_stations_weather():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    mbajk_dir = os.path.join(base_dir, 'data/raw/mbajk')

    for filename in os.listdir(mbajk_dir):
        file_path = os.path.join(mbajk_dir, filename)
        df = pd.read_csv(file_path)
        lat = df['lat'].iloc[0]
        lng = df['lng'].iloc[0]
        weather = fetch_weather(lat, lng)
        weather_df = pd.DataFrame(weather['hourly'])

        expected_columns = ['date', 'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
                            'precipitation_probability', 'rain', 'surface_pressure']

        for index, row in weather_df.iterrows():
            if row.tolist() == expected_columns:
                weather_df.drop(index, inplace=True)

        weather_df.rename(columns={
            'time': 'date',
            'temperature_2m': 'temperature',
            'relative_humidity_2m': 'relative_humidity',
            'dew_point_2m': 'dew_point'
        }, inplace=True)
        weather_file_path = f'{base_dir}/data/raw/weather/{filename}'
        weather_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        weather_df.to_csv(weather_file_path, mode='a', index=False)

    remove_header_rows(f'{base_dir}/data/raw/weather/')


if __name__ == '__main__':
    fetch_stations_weather()
