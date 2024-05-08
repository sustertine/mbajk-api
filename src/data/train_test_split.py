import os

import pandas as pd


def split_data():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')

    for file in os.listdir(f'{base_dir}/data/processed/mbajk'):
        df = pd.read_csv(f'{base_dir}/data/processed/mbajk/{file}')
        split = int(len(df) * 0.9)

        train = df.iloc[:split]
        test = df.iloc[split:]

        filename = os.path.splitext(file)[0]
        if not os.path.exists(f'{base_dir}/data/processed/train_test/{filename}'):
            os.makedirs(f'{base_dir}/data/processed/train_test/{filename}')

        train.to_csv(f'{base_dir}/data/processed/train_test/{filename}/train.csv', index=False)
        test.to_csv(f'{base_dir}/data/processed/train_test/{filename}/test.csv', index=False)


if __name__ == '__main__':
    split_data()
