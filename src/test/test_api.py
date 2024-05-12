import pytest
from starlette.testclient import TestClient
from src.serve.main import app


def test_predict_api():
    client = TestClient(app)

    url = "/api/health"

    response = client.get(url)

    assert response.status_code == 200

    assert response.json() == {"status": "ok"}
