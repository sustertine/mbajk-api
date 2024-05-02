import os
import pandas as pd


def read_station_data(file_path):
    df = pd.read_csv(file_path)
    return df[['name', 'lat', 'lng']].drop_duplicates().to_dict('records')


def get_stations_locations():
    cwd = os.getcwd()

    directory = os.path.join(cwd, 'data', 'raw', 'mbajk')

    all_stations = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            all_stations.extend(read_station_data(file_path))
    return all_stations
