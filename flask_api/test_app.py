import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_healthcheck_endpoint(client):
    response = client.get('/healthcheck')
    assert response.status_code == 200
    assert response.json == {"status": "healthy"}
