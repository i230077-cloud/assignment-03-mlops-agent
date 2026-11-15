"""
Unit Tests for FastAPI Model Inference Application & Probes
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "customer-churn-inference"
    assert "endpoints" in data

def test_liveness_probe(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "uptime_seconds" in data

def test_readiness_probe(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True

def test_version_endpoint(client):
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["app_version"] == "v1.2.0"
    assert data["model_name"] == "CustomerChurnModel"

def test_predict_endpoint_valid_payload(client):
    payload = {
        "customer_id": "CUST_9999",
        "tenure_months": 24,
        "monthly_charges": 85.50,
        "total_charges": 2052.0,
        "support_calls": 2,
        "contract_type": "month-to-month",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST_9999"
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert data["inference_latency_ms"] > 0

def test_predict_endpoint_invalid_payload(client):
    bad_payload = {
        "customer_id": "CUST_BAD",
        "tenure_months": -5,  # validation constraint violated
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
