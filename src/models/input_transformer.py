import pandas as pd

from src.models.input_model import InputModel


class InputTransformer:
    def transform(self, data):
        data = data.dict()
        data.pop('station_name')
        data = {k: [v] for k, v in data.items()}

        df = pd.DataFrame(data)

        df['date'] = pd.to_datetime('today').date()

        return df
