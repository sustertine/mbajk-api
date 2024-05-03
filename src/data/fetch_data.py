import os
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    url = os.getenv('MBAJK_URL')
    response = requests.get(url)

    df = pd.read_json(StringIO(response.text))
    df.drop(columns=['contract_name', 'number'], inplace=True, axis=1)
    df['last_update'] = pd.to_datetime(df['last_update'], unit='ms')

    df_position = pd.json_normalize(df['position'])
    df = pd.concat([df, df_position], axis=1)
    df.drop(columns=['position'], inplace=True)

    for name, group in df.groupby('name'):
        # filename = f'../../data/raw/mbajk/{name}.csv'
        # filename = os.path.abspath(f'../../data/raw/mbajk/{name}.csv')
        filename = os.path.join(os.getenv('GITHUB_WORKSPACE'), 'data', 'raw', 'mbajk', f'{name}.csv')
        if os.path.join(filename):
            existing_df = pd.read_csv(filename, parse_dates=['last_update'])
            if existing_df['last_update'].max() < group['last_update'].max():
                group.to_csv(filename, mode='a', header=False, index=False)
        else:
            group.to_csv(filename, index=False)

