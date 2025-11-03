from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any
import os, json
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

app = FastAPI(title="CIRFOOD Forecast Service", version="1.0.0")

DATA_DIR = os.getenv("DATA_DIR", "./data")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
FREQUENCY = os.getenv("FREQUENCY", "W")
HORIZON = int(os.getenv("HORIZON", 8))
SEASONAL_PERIODS = 52 if FREQUENCY.upper().startswith("W") else 12
API_KEY = os.getenv("API_KEY", None)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class RunRequest(BaseModel):
    horizon: int | None = None
    frequency: str | None = None  # 'W' o 'M'

class ForecastItem(BaseModel):
    categoria: str
    ds: str
    yhat: float
    yhat_lower: float | None = None
    yhat_upper: float | None = None

def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.get("/")
def root():
    return {
        "message": "Benvenuto nella CIRFOOD Forecast API! Il servizio è attivo.",
        "endpoints": ["/health", "/jobs/run", "/forecasts/latest", "/model-registry"]
    }