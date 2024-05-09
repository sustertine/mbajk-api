import os

import numpy as np
import pandas as pd
import concurrent.futures
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.optimizers import Adam
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.feature_engineer import FeatureEngineer


def create_dataset(dataset, look_back=1, look_forward=5):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward):
        a = dataset.iloc[i:(i + look_back), :]
        dataX.append(a)

        dataY.append(dataset.iloc[i + look_back:i + look_back + look_forward]['available_bike_stands'])
    return np.array(dataX), np.array(dataY)


def train_model(file_path):
    df = pd.read_csv(file_path)


    station_name = os.path.splitext(os.path.basename(file_path))[0]

    look_back = 1
    look_forward = 7
    dataX, dataY = create_dataset(df, look_back, look_forward)

    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(dataX.shape[1], dataX.shape[2]), return_sequences=True))
    model.add(Dense(100, activation='relu'))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(7))
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(),
                  metrics=[MeanSquaredError(), MeanAbsoluteError()])
    model.summary()

    history = model.fit(dataX, dataY, epochs=100, batch_size=16, verbose=1)
    model.save(f'../../models/{station_name}.h5')


def train_models():
    csv_files = [os.path.join(os.path.abspath("../../data/processed/mbajk"), file) for file in
                 os.listdir("../../data/processed/mbajk") if file.endswith(".csv")]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(train_model, csv_files)


if __name__ == '__main__':
    train_models()