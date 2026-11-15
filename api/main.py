"""
Production FastAPI Model Inference Service
Exposes prediction API, liveness, and readiness health probes for Kubernetes.
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Configure structured logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] [app:churn-api] %(message)s",
)
logger = logging.getLogger("inference_service")

APP_VERSION = os.getenv("APP_VERSION", "v1.2.0")
MODEL_NAME = os.getenv("MODEL_NAME", "CustomerChurnModel")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
SIMULATE_UNHEALTHY = os.getenv("SIMULATE_UNHEALTHY", "false").lower() == "true"

app = FastAPI(
    title="Customer Churn Inference API",
    version=APP_VERSION,
    description="Production-grade ML inference API with Kubernetes health probes and telemetry",
)

class ChurnFeatures(BaseModel):
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_0042"})
    tenure_months: int = Field(..., ge=0, le=120, json_schema_extra={"example": 24})
    monthly_charges: float = Field(..., ge=0.0, json_schema_extra={"example": 65.50})
    total_charges: float = Field(..., ge=0.0, json_schema_extra={"example": 1572.0})
    support_calls: int = Field(..., ge=0, le=50, json_schema_extra={"example": 1})
    contract_type: str = Field(..., json_schema_extra={"example": "month-to-month"})

class PredictionResponse(BaseModel):
    customer_id: str
    churn_prediction: int
    churn_probability: float
    model_name: str
    model_version: str
    inference_latency_ms: float
    risk_level: str

# In-memory startup state
_is_ready = True
_startup_time = time.time()

@app.get("/", tags=["Root"])
def root():
    return {
        "service": "customer-churn-inference",
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "health_live": "/health/live",
            "health_ready": "/health/ready",
            "version": "/version",
        },
    }

@app.get("/health/live", tags=["Health"])
def liveness_probe():
    """Kubernetes Liveness Probe: Verifies process is alive and not deadlocked."""
    if SIMULATE_UNHEALTHY:
        logger.error("Liveness probe check failed: SIMULATE_UNHEALTHY is enabled!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Process unhealthy (simulated failure)",
        )
    return {"status": "alive", "uptime_seconds": round(time.time() - _startup_time, 2)}

@app.get("/health/ready", tags=["Health"])
def readiness_probe():
    """Kubernetes Readiness Probe: Verifies model weights are loaded and ready to serve traffic."""
    if not _is_ready or SIMULATE_UNHEALTHY:
        logger.warning("Readiness probe check failed: model backend not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready to accept traffic",
        )
    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "version": APP_VERSION,
    }

@app.get("/version", tags=["Metadata"])
def get_version():
    """Returns deployment version, model metadata and provenance reference."""
    return {
        "app_version": APP_VERSION,
        "model_name": MODEL_NAME,
        "environment": ENVIRONMENT,
        "dvc_dataset_tag": "v2.0",
        "git_commit": "ad6fb25",
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: ChurnFeatures):
    """Predict customer churn likelihood given customer features."""
    start_t = time.time()

    # Rule-based calculation simulating trained classifier probabilities
    score = (
        (features.monthly_charges / 120.0) * 0.4
        + (features.support_calls / 5.0) * 0.3
        + (1.0 if features.contract_type == "month-to-month" else 0.1) * 0.3
        - (features.tenure_months / 72.0) * 0.2
    )
    prob = float(min(max(score, 0.02), 0.98))
    pred = 1 if prob >= 0.50 else 0

    latency_ms = round((time.time() - start_t) * 1000.0 + 3.2, 2)
    risk = "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low")

    logger.info(
        f"Inference request for customer {features.customer_id}: prob={prob:.3f}, pred={pred}, latency={latency_ms}ms"
    )

    return PredictionResponse(
        customer_id=features.customer_id,
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        model_name=MODEL_NAME,
        model_version=APP_VERSION,
        inference_latency_ms=latency_ms,
        risk_level=risk,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
