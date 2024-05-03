import pandas as pd
import requests
import os


def fetch_weather(lat, lng):
    url = (f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}1&hourly=temperature_2m,'
           f'relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure')
    return requests.get(url).json()


def fetch_stations_weather():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../data/raw/mbajk')
    print(f'\nFetching weather for stations in {base_dir}\n')
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        print(f'Processing {file_path}')
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                print(f'Fetching weather for {filename}')
                print('19th line:')
                print(lines[18])
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
            weather_file_path = f'../../data/raw/weather/{filename}'
            weather_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
            weather_df.to_csv(weather_file_path, mode='a', index=False)


if __name__ == '__main__':
    fetch_stations_weather()
