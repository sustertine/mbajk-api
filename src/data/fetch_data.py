from io import StringIO

import pandas as pd
import requests

if __name__ == '__main__':
    url = 'https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b'

    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_json(StringIO(response.text))
        print(df.shape)
        # df.to_csv('data/raw/maribor.csv', index=False)
    else:
        print('Error:', response.status_code)

