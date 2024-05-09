import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam
import mlflow
from scipy.stats import yeojohnson
from sklearn.preprocessing import MinMaxScaler, StandardScaler

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')


def preprocess_data(df):
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    station_name = df['name'].iloc[0]

    df.drop(columns=['name'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    for column in df.drop(columns=['date'], axis=1).columns:
        df[column], _ = yeojohnson(df[column])

    targetScaler = MinMaxScaler()
    std = StandardScaler()
    mms = MinMaxScaler()

    df['available_bike_stands'] = targetScaler.fit_transform(
        df['available_bike_stands'].values.reshape(-1, 1))
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

    # join base data, commented out for speed of fitting
    # df_base = pd.read_csv(f'{base_dir}/data/processed/mbajk_dataset.csv')
    # merged_df['date'] = pd.to_datetime(merged_df['date']).dt.tz_localize(None)
    # df_base['date'] = pd.to_datetime(df_base['date']).dt.tz_localize(None)
    # merged_df = pd.concat([df_base, merged_df], axis=0)

    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

    df = df[['date', 'temperature', 'dew_point', 'apparent_temperature',
                           'surface_pressure', 'available_bike_stands',
                           'lagged_available_bike_stands', 'rolling_mean_bike_stands',
                           'rolling_std_bike_stands', 'diff_available_bike_stands',
                           'temperature_diff']]

    df.sort_values(by='date', inplace=True)
    return df

def create_dataset(dataset, look_back=1, look_forward=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward):
        a = dataset.iloc[i:(i + look_back), :]
        dataX.append(a)

        dataY.append(dataset.iloc[i + look_back:i + look_back + look_forward]['available_bike_stands'])
    return np.array(dataX), np.array(dataY)


def train_eval_model(train_file_path, test_file_path):
    station_name = os.path.basename(os.path.dirname(train_file_path))

    print(f'Training model for station: {station_name}')

    mlflow.set_experiment('evaluate models')
    with mlflow.start_run(run_name=station_name):
        train_df = pd.read_csv(train_file_path)
        train_df.drop('date', axis=1, inplace=True)

        test_df = pd.read_csv(test_file_path)
        test_df.drop('date', axis=1, inplace=True)

        look_back = 1
        look_forward = 7
        trainX, trainY = create_dataset(train_df, look_back, look_forward)
        testX, testY = create_dataset(test_df, look_back, look_forward)

        hidden_size = 100
        learning_rate = 0.001
        batch_size = 16
        n_epochs = 50

        mlflow.tensorflow.autolog()

        model = Sequential()
        model.add(
            LSTM(hidden_size, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(LSTM(hidden_size, activation='relu'))
        model.add(Dense(7))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(),
                      metrics=[MeanSquaredError(), MeanAbsoluteError()])

        history = model.fit(trainX, trainY, epochs=n_epochs, batch_size=batch_size, verbose=1)

        evaluation = model.evaluate(testX, testY, verbose=1)

def train_eval_models():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    tt_dir = f'{base_dir}/data/processed/train_test'
    subdirectories = [os.path.join(tt_dir, o) for o in os.listdir(tt_dir) if os.path.isdir(os.path.join(tt_dir, o))]

    for subdir in subdirectories:
        train_file = os.path.join(subdir, 'train.csv')
        test_file = os.path.join(subdir, 'test.csv')
        train_eval_model(train_file, test_file)


if __name__ == '__main__':
    train_eval_models()