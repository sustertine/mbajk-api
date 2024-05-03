import pandas as pd
import requests
import os


def fetch_weather(lat, lng):
    url = (f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}1&hourly=temperature_2m,'
           f'relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure')
    return requests.get(url).json()


def fetch_stations_weather():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../data/raw/mbajk')
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        lat = df['lat'].iloc[0]
        lng = df['lng'].iloc[0]
        weather = fetch_weather(lat, lng)
        weather_df = pd.DataFrame(weather['hourly'])

        expected_columns = ['date', 'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
                            'precipitation_probability', 'rain', 'surface_pressure']

        if weather_df.iloc[0].tolist() == expected_columns:
            weather_df = weather_df.iloc[1:]

        weather_df.rename(columns={
            'time': 'date',
            'temperature_2m': 'temperature',
            'relative_humidity_2m': 'relative_humidity',
            'dew_point_2m': 'dew_point'
        }, inplace=True)
        weather_file_path = f'../../data/raw/weather/{filename}'
        weather_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        weather_df.to_csv(weather_file_path, mode='a', index=False)


if __name__ == '__main__':
    fetch_stations_weather()
