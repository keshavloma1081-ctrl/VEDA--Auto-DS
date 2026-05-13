"""API Integration Tests"""
import pytest
from fastapi.testclient import TestClient
import os

os.environ['GROQ_API_KEY'] = 'test_key'

from veda.api.server import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "VEDA ML API"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "components" in response.json()

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    assert "workflows" in response.json()

def test_login_endpoint():
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_list_workflows():
    response = client.get("/workflows")
    assert response.status_code == 200
    assert isinstance(response.json(), list)