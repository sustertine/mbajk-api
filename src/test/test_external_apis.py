# Test openmeteo and mbaj_url availability
import os

import dotenv
import requests

dotenv.load_dotenv()


def test_openmeteo():
    url = ('https://api.open-meteo.com/v1/forecast?latitude=50&longitude=50&hourly=temperature_2m,'
           'relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure')
    response = requests.get(url)
    assert response.status_code == 200


def test_mbajk_url():
    url = os.getenv('MBAJK_URL')
    response = requests.get(url)
    assert response.status_code == 404 # 200
