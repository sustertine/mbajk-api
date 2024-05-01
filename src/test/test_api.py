import pytest
from starlette.testclient import TestClient
from src.serve.main import app


def test_predict_api():
    client = TestClient(app)

    url = "/api/mbajk/predict"

    payload = {
        "temperature": 20.0,
        "dew_point": 10.0,
        "apparent_temperature": 18.0,
        "surface_pressure": 1000.0,
        "available_bike_stands": 5,
        "lagged_available_bike_stands": 0.579272,
        "rolling_mean_bike_stands": 0.579318,
        "rolling_std_bike_stands": 0.049509,
        "diff_available_bike_stands": 0.000007,
        "temperature_diff": -5.458696e-16,
        "rain": 0.0
    }

    response = client.post(url, json=payload)

    assert response.status_code == 200

    assert "prediction" in response.json()
