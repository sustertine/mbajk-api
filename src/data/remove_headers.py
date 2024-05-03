import os
import pandas as pd


def remove_header_rows(directory):
    expected_columns = ['date', 'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
                        'precipitation_probability', 'rain', 'surface_pressure']

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            for index, row in df.iterrows():
                if row.tolist() == expected_columns:
                    df.drop(index, inplace=True)

            df.to_csv(file_path, index=False)


if __name__ == '__main__':
    remove_header_rows('../../data/raw/weather')
