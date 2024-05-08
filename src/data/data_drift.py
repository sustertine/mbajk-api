import os

import pandas as pd
import dvc.api

def create_current_data():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')

    df_current = pd.DataFrame()

    for filename in os.listdir(f'{base_dir}/data/processed/mbajk'):
        df_station = pd.read_csv(f'{base_dir}/data/processed/mbajk/{filename}')
        df_current = pd.concat([df_current, df_station])

    df_current.to_csv(f'{base_dir}/data/processed/current_data.csv', index=False)

def create_reference_data():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    df_reference = pd.DataFrame()

    with dvc.api.open(
            repo=base_dir,
            path=f'{base_dir}/data/processed/mbajk',
            rev='HEAD~1'
    ) as reference_data:
        reference_csvs = [f for f in reference_data.dvcfiles if f.endswith('.csv')]

    for reference_csv in reference_csvs:
        df_station = pd.read_csv(reference_csv)
        df_reference = pd.concat([df_reference, df_station])

    df_reference.to_csv(f'{base_dir}/data/processed/reference_data.csv', index=False)


def data_drift():
    pass

if __name__ == '__main__':
    # create_current_data()
    create_reference_data()
