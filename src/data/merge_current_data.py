import os

import pandas as pd


def merge_current_data():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    df_current = pd.DataFrame()
    for filename in os.listdir(f'{base_dir}/data/processed/mbajk'):
        file_path = os.path.join(f'{base_dir}/data/processed/mbajk', filename)
        df = pd.read_csv(file_path)
        df_current = pd.concat([df_current, df])

    df_current.to_csv(f'{base_dir}/data/processed/current_data.csv', index=False)

if __name__ == '__main__':
    merge_current_data()