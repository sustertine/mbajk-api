import os

import pandas as pd
from scipy.stats import yeojohnson
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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

        yield merged_df


def preprocess_data(merged_df):
    station_name = merged_df['name'].iloc[0]

    merged_df.drop(columns=['name'], inplace=True)
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df.sort_values(by='date', inplace=True)

    for column in merged_df.drop(columns=['date'], axis=1).columns:
        merged_df[column], _ = yeojohnson(merged_df[column])

    targetScaler = MinMaxScaler()
    std = StandardScaler()
    mms = MinMaxScaler()

    merged_df['available_bike_stands'] = targetScaler.fit_transform(
        merged_df['available_bike_stands'].values.reshape(-1, 1))
    merged_df[['temperature', 'dew_point', 'apparent_temperature', 'surface_pressure']] = std.fit_transform(
        merged_df[['temperature', 'dew_point', 'apparent_temperature', 'surface_pressure']])
    merged_df[['relative_humidity', 'precipitation_probability', 'rain']] = mms.fit_transform(
        merged_df[['relative_humidity', 'precipitation_probability', 'rain']])

    merged_df['time_of_day'] = merged_df['date'].dt.hour // 6
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
    merged_df = pd.get_dummies(merged_df, columns=['time_of_day', 'day_of_week'], drop_first=True)

    merged_df['lagged_available_bike_stands'] = merged_df['available_bike_stands'].shift(1)
    lagged_mean = merged_df['lagged_available_bike_stands'].mean()
    merged_df['lagged_available_bike_stands'].fillna(lagged_mean, inplace=True)

    window_size = 7
    merged_df['rolling_mean_bike_stands'] = merged_df['available_bike_stands'].rolling(window=window_size).mean()
    rolled_mean = merged_df['rolling_mean_bike_stands'].mean()
    merged_df['rolling_mean_bike_stands'].fillna(rolled_mean, inplace=True)

    merged_df['rolling_std_bike_stands'] = merged_df['available_bike_stands'].rolling(window=window_size).std()
    rolled_std_mean = merged_df['rolling_std_bike_stands'].mean()
    merged_df['rolling_std_bike_stands'].fillna(rolled_std_mean, inplace=True)

    merged_df['diff_available_bike_stands'] = merged_df['available_bike_stands'].diff()
    diff_mean = merged_df['diff_available_bike_stands'].mean()
    merged_df['diff_available_bike_stands'].fillna(diff_mean, inplace=True)

    merged_df['temperature_diff'] = merged_df['temperature'] - merged_df['apparent_temperature']
    temperature_diff_mean = merged_df['temperature_diff'].mean()
    merged_df['temperature_diff'].fillna(temperature_diff_mean, inplace=True)

    df_base = pd.read_csv('../../data/processed/mbajk_dataset.csv')
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.tz_localize(None)
    df_base['date'] = pd.to_datetime(df_base['date']).dt.tz_localize(None)
    merged_df = pd.concat([df_base, merged_df], axis=0)

    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.tz_localize(None)

    merged_df = merged_df[['date', 'temperature', 'dew_point', 'apparent_temperature',
                           'surface_pressure', 'available_bike_stands',
                           'lagged_available_bike_stands', 'rolling_mean_bike_stands',
                           'rolling_std_bike_stands', 'diff_available_bike_stands',
                           'temperature_diff']]

    merged_df.sort_values(by='date', inplace=True)

    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    merged_df.to_csv(f'{base_dir}/data/processed/mbajk/{station_name}.csv', index=False)


if __name__ == '__main__':
    for df in merge_weather_stations():
        preprocess_data(df)
