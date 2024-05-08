import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam
import mlflow

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')


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
        n_epochs = 3

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