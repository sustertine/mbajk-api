import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam
import mlflow
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import set_config

set_config(transform_output="pandas")

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['date'] = pd.to_datetime(X['date'])
        X.sort_values(by='date', inplace=True)
        X['time_of_day'] = X['date'].dt.hour // 6

        X['day_of_week'] = X['date'].dt.dayofweek
        X = pd.get_dummies(X, columns=['time_of_day', 'day_of_week'], drop_first=True)

        X['lagged_available_bike_stands'] = X['available_bike_stands'].shift(1)

        window_size = 7
        X['rolling_mean_bike_stands'] = X['available_bike_stands'].rolling(window=window_size).mean()

        X['rolling_std_bike_stands'] = X['available_bike_stands'].rolling(window=window_size).std()

        X['diff_available_bike_stands'] = X['available_bike_stands'].diff()

        X['temperature_diff'] = X['temperature'] - X['apparent_temperature']

        X = X[['temperature', 'dew_point', 'apparent_temperature',
               'surface_pressure', 'available_bike_stands',
               'lagged_available_bike_stands', 'rolling_mean_bike_stands',
               'rolling_std_bike_stands', 'diff_available_bike_stands',
               'temperature_diff']]

        return X


def create_dataset(dataset, look_back=1, look_forward=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward):
        a = dataset.iloc[i:(i + look_back), :]
        dataX.append(a)

        dataY.append(dataset.iloc[i + look_back:i + look_back + look_forward]['available_bike_stands'])
    return np.array(dataX), np.array(dataY)


def preprocess_data(df):
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    standard_features = ['temperature', 'dew_point', 'apparent_temperature', 'surface_pressure']
    minmax_features = ['lagged_available_bike_stands',
                       'rolling_mean_bike_stands',
                       'rolling_std_bike_stands', 'diff_available_bike_stands',
                       'temperature_diff']
    target_feature = ['available_bike_stands']

    column_transformer = ColumnTransformer(
        transformers=[
            ('standard_scaler', standard_scaler, standard_features),
            ('minmax_scaler', minmax_scaler, minmax_features),
        ]
    )

    pipeline = Pipeline(steps=[
        ('feature_engineering', FeatureEngineer()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('column_transformer', column_transformer),
    ])

    out_df = pipeline.fit_transform(df)

    target = target_scaler.fit_transform(df[target_feature].values.reshape(-1, 1))
    # export the target scaler
    # export pipeline
    out_df[target_feature] = target

    return out_df


def train_eval_model(train_file_path, test_file_path):
    station_name = os.path.basename(os.path.dirname(train_file_path))

    print(f'Training model for station: {station_name}')

    mlflow.set_experiment('evaluate models')
    with mlflow.start_run(run_name=station_name):
        train_df = pd.read_csv(train_file_path)
        test_df = pd.read_csv(test_file_path)
        test_df = preprocess_data(test_df)
        train_df = preprocess_data(train_df)

        look_back = 1
        look_forward = 7
        trainX, trainY = create_dataset(train_df, look_back, look_forward)
        testX, testY = create_dataset(test_df, look_back, look_forward)

        hidden_size = 100
        learning_rate = 0.001
        batch_size = 16
        n_epochs = 2

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
