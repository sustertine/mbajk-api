import os

import pandas as pd
from scipy.stats import yeojohnson
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def merge_weather_stations():
    base_dir = os.path.abspath('../../data/raw/mbajk')
    weather_dir = os.path.abspath('../../data/raw/weather')
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        weather_file_path = os.path.join(weather_dir, filename)
        weather_df = pd.read_csv(weather_file_path)

        df['last_update'] = pd.to_datetime(df['last_update'])
        weather_df['date'] = pd.to_datetime(weather_df['date'])

        df['date_hour'] = df['last_update'].dt.to_period('h')
        weather_df['date_hour'] = weather_df['date'].dt.to_period('h')

        merged_df = pd.merge(df, weather_df, on='date_hour', how='inner')
        merged_df.drop(
            columns=['date_hour', 'address', 'banking', 'bonus', 'lat', 'lng', 'bike_stands', 'date', 'status'],
            inplace=True)

        merged_df.rename(columns={'last_update': 'date'}, inplace=True)
        print(merged_df.isnull().sum())

        yield merged_df


def preprocess_data(df):
    station_name = df['name'].iloc[0]

    df.drop(columns=['name'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    for column in df.drop(columns=['date'], axis=1).columns:
        df[column], _ = yeojohnson(df[column])

    targetScaler = MinMaxScaler()
    std = StandardScaler()
    mms = MinMaxScaler()

    df['available_bike_stands'] = targetScaler.fit_transform(df['available_bike_stands'].values.reshape(-1, 1))
    df[['temperature', 'dew_point', 'apparent_temperature', 'surface_pressure']] = std.fit_transform(
        df[['temperature', 'dew_point', 'apparent_temperature', 'surface_pressure']])
    df[['relative_humidity', 'precipitation_probability', 'rain']] = mms.fit_transform(
        df[['relative_humidity', 'precipitation_probability', 'rain']])

    df['time_of_day'] = df['date'].dt.hour // 6
    df['day_of_week'] = df['date'].dt.dayofweek
    df = pd.get_dummies(df, columns=['time_of_day', 'day_of_week'], drop_first=True)

    df['lagged_available_bike_stands'] = df['available_bike_stands'].shift(1)
    lagged_mean = df['lagged_available_bike_stands'].mean()
    df['lagged_available_bike_stands'].fillna(lagged_mean, inplace=True)

    window_size = 7
    df['rolling_mean_bike_stands'] = df['available_bike_stands'].rolling(window=window_size).mean()
    rolled_mean = df['rolling_mean_bike_stands'].mean()
    df['rolling_mean_bike_stands'].fillna(rolled_mean, inplace=True)

    df['rolling_std_bike_stands'] = df['available_bike_stands'].rolling(window=window_size).std()
    rolled_std_mean = df['rolling_std_bike_stands'].mean()
    df['rolling_std_bike_stands'].fillna(rolled_std_mean, inplace=True)

    df['diff_available_bike_stands'] = df['available_bike_stands'].diff()
    diff_mean = df['diff_available_bike_stands'].mean()
    df['diff_available_bike_stands'].fillna(diff_mean, inplace=True)

    df['temperature_diff'] = df['temperature'] - df['apparent_temperature']
    temperature_diff_mean = df['temperature_diff'].mean()
    df['temperature_diff'].fillna(temperature_diff_mean, inplace=True)

    df = df[['date', 'temperature', 'dew_point', 'apparent_temperature',
             'surface_pressure', 'available_bike_stands',
             'lagged_available_bike_stands', 'rolling_mean_bike_stands',
             'rolling_std_bike_stands', 'diff_available_bike_stands',
             'temperature_diff']]
    print(df.describe())

    df.to_csv(f'../../data/processed/{station_name}.csv', index=False)


if __name__ == '__main__':
    for df in merge_weather_stations():
        preprocess_data(df)
        break
