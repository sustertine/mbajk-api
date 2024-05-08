import os

import pandas as pd


def split_data():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    df = pd.read_csv(f'{base_dir}/data/processed/current_data.csv')

    split = int(len(df) * 0.9)

    train = df.iloc[:split]
    test = df.iloc[split:]

    train.to_csv(f'{base_dir}/data/processed/train.csv', index=False)
    test.to_csv(f'{base_dir}/data/processed/test.csv', index=False)


if __name__ == '__main__':
    split_data()
