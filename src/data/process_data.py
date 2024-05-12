import os

import pandas as pd


def find_nearest(row, df, column='date'):
    exactmatch = df[df[column] == row['last_update']]
    if not exactmatch.empty:
        return exactmatch.iloc[0]
    else:
        lower = df[df[column] < row['last_update']].max()
        upper = df[df[column] > row['last_update']].min()
        nearest = lower if (row['last_update'] - lower[column]) < (upper[column] - row['last_update']) else upper
        return nearest


def merge_weather_stations():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    mbajk_dir = os.path.abspath(f'{base_dir}/data/raw/mbajk')
    weather_dir = os.path.abspath(f'{base_dir}/data/raw/weather')
    for filename in os.listdir(mbajk_dir):
        file_path = os.path.join(mbajk_dir, filename)
        station_df = pd.read_csv(file_path)

        weather_file_path = os.path.join(weather_dir, filename)
        weather_df = pd.read_csv(weather_file_path)

        station_df['last_update'] = pd.to_datetime(station_df['last_update'])
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        merged_df = station_df.apply(find_nearest, args=(weather_df,), axis=1)
        merged_df = pd.concat([station_df, merged_df], axis=1)

        merged_df.drop(
            columns=['address', 'banking', 'bonus', 'lat', 'lng', 'bike_stands', 'status', 'date'],
            inplace=True)

        merged_df.rename(columns={'last_update': 'date'}, inplace=True)
        station_name = merged_df['name'].iloc[0]
        merged_df.drop(columns=['name'], inplace=True)
        merged_df.to_csv(f'{base_dir}/data/processed/mbajk/{station_name}.csv', index=False)

if __name__ == '__main__':
    merge_weather_stations()
