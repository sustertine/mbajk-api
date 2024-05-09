import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
set_config(transform_output = "pandas")

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