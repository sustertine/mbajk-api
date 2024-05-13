import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from datetime import datetime, timedelta
import mlflow

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')

def main():
    print(os.getenv('MONGO_URL'))
    client = MongoClient(os.getenv('MONGO_URL'))

    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    mbajk_dir = os.path.join(base_dir, 'data/raw/mbajk')  # raw still has date

    api_calls = client['mbajk-api-calls']['mbajk-api-calls']

    mlflow.set_experiment('evaluate prod predictions')
    for dataset in os.listdir(mbajk_dir):
        with mlflow.start_run(run_name=dataset.split('.')[0]):
            df = pd.read_csv(os.path.join(mbajk_dir, dataset))
            dataset_api_calls = api_calls.find({'station_name': {'$regex': dataset.split('.')[0]}})

            predictions_by_date = []
            for call in dataset_api_calls:
                predictions_by_date.append(get_prediction_on_date(call))

            station_predictions = [item for sub_list in predictions_by_date for item in sub_list]

            df['last_update'] = pd.to_datetime(df['last_update'])
            mse, mae, evs = calculate_errors(df, station_predictions)
            mlflow.log_param('dataset', os.path.join(mbajk_dir, dataset))
            print(f'{dataset}:\n\t- MSE: {mse}\n\t- MAE: {mae}\n\t- EVS: {evs}')

            if mse == 'NaN':
                mlflow.log_metrics({'mse': 0, 'mae': 0, 'evs': 0})
                continue
            else:
                mlflow.log_metrics({'mse': mse, 'mae': mae, 'evs': evs})


def closest_date(df, target_date):
    closest_idx = df.index[(df['last_update'] - target_date).abs().argmin()]
    return df.loc[closest_idx]


def calculate_errors(df, station_predictions):
    y_true = []
    y_pred = []

    if not station_predictions:
        return 'NaN', 'NaN', 'NaN'

    for item in station_predictions:
        item_date = datetime.fromisoformat(item['date'])

        closest_row = closest_date(df, item_date)

        y_true.append(closest_row['available_bike_stands'])
        y_pred.append(item['prediction'])

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    return mse, mae, evs


def get_prediction_on_date(api_call):
    date = datetime.fromisoformat(api_call['date'])

    predictions = []
    for i, prediction in enumerate(api_call['predictions']):
        new_date = date + timedelta(hours=i)
        predictions.append({
            'date': str(new_date),
            'prediction': prediction
        })

    return predictions


if __name__ == '__main__':
    main()
