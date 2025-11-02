from __future__ import annotations
import os, json
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

APP_VERSION = "1.0.0"
DATA_DIR = os.environ.get("DATA_DIR", "./data")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")
FREQUENCY = os.environ.get("FREQUENCY", "W")
HORIZON = int(os.environ.get("HORIZON", 8))
SEASONAL_PERIODS = 52 if FREQUENCY.upper().startswith("W") else 12
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CATEGORIA_TO_MODEL = {
    "primo piatto": "ETS",
    "secondo piatto": "PROPHET",
    "frutta": "SARIMAX",
    "dolce": "CROSTON_SBA",
}

class RunRequest(BaseModel):
    horizon: int | None = None
    frequency: str | None = None

class ForecastItem(BaseModel):
    categoria: str
    ds: str
    yhat: float
    yhat_lower: float | None = None
    yhat_upper: float | None = None

app = FastAPI(title="CIRFOOD Forecast Service", version=APP_VERSION)

CANON_COLS = ["regione", "città", "scuola", "mese", "settimana", "categoria piatto", "piatto", "valore"]
ALIASES = {
    "regione": ["regione", "region"],
    "città": ["città", "citta", "city"],
    "scuola": ["scuola", "school"],
    "mese": ["mese", "month"],
    "settimana": ["settimana", "week"],
    "categoria piatto": ["categoria piatto", "categoria", "categoria_cibo", "categoria_piatto"],
    "piatto": ["piatto", "dish"],
    "valore": ["valore", "quantita", "qty", "quantity", "volume"],
}

def _read_csv_any(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    rename_map = {}
    for canon, alts in ALIASES.items():
        for c in df.columns:
            if c in alts:
                rename_map[c] = canon
    df.rename(columns=rename_map, inplace=True)
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = np.nan if c in ["mese", "settimana"] else None
    if "valore" in df.columns:
        df["valore"] = pd.to_numeric(df["valore"], errors="coerce").fillna(0)
        df.loc[df["valore"] < 0, "valore"] = 0
    return df[CANON_COLS]

def load_base() -> pd.DataFrame:
    vendite = _read_csv_any(os.path.join(DATA_DIR, "vendite.csv"))
    scarto_teglia = _read_csv_any(os.path.join(DATA_DIR, "scarto_teglia.csv"))
    scarto_piatto = _read_csv_any(os.path.join(DATA_DIR, "scarto_piatto.csv"))

    keys = ["regione", "città", "scuola", "mese", "settimana", "categoria piatto", "piatto"]
    df = vendite.merge(scarto_teglia, on=keys, how="outer", suffixes=("_vendite", "_teglia"))
    df = df.merge(scarto_piatto, on=keys, how="outer")
    df.rename(columns={"valore": "valore_piatto"}, inplace=True)

    for col in ["valore_vendite", "valore_teglia", "valore_piatto"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).clip(lower=0)

    df["consumo_netto"] = (df["valore_vendite"] - df["valore_teglia"] - df["valore_piatto"]).clip(lower=0)

    if FREQUENCY.upper().startswith("W"):
        def row_to_date(r):
            try:
                week = int(r.get("settimana") or 1)
                year = datetime.today().year
                return datetime.fromisocalendar(year, min(max(week,1), 53), 1)
            except:
                return pd.NaT
        df["ds"] = df.apply(row_to_date, axis=1)
    else:
        df["ds"] = pd.to_datetime(df["mese"], errors="coerce")
        df["ds"].fillna(pd.to_datetime(df["mese"].astype(str) + "-01", errors="coerce"), inplace=True)

    df.dropna(subset=["ds"], inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    agg = df.groupby(["categoria piatto", "ds"], dropna=True)["consumo_netto"].sum().reset_index()
    agg.rename(columns={"categoria piatto": "categoria", "consumo_netto": "y"}, inplace=True)

    out = []
    for cat, g in agg.groupby("categoria"):
        g = g.sort_values("ds").set_index("ds")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq=FREQUENCY)
        g = g.reindex(full_idx).fillna(0)
        g["categoria"] = cat
        out.append(g.reset_index().rename(columns={"index": "ds"}))
    return pd.concat(out, ignore_index=True)

def fit_ets(y): return ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=SEASONAL_PERIODS, damped_trend=True).fit(smoothing_level=0.4, smoothing_slope=0.2, smoothing_seasonal=0.2, optimized=False)
def fit_sarimax(y): return SARIMAX(y, order=(0,1,1), seasonal_order=(1,1,0, SEASONAL_PERIODS), trend='c', enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
def fit_prophet(df):
    if not _HAS_PROPHET:
        raise RuntimeError("Prophet non disponibile.")
    m = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.10, seasonality_prior_scale=10, holidays_prior_scale=10)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    m.fit(df.rename(columns={"ds":"ds", "y":"y"})[["ds","y"]])
    return m
def croston_sba_forecast(y, h, alpha=0.2):
    y = y.fillna(0).values
    demand, intervals, gap = [], [], 0
    for val in y:
        if val > 0:
            demand.append(val)
            intervals.append(gap or 1)
            gap = 1
        else:
            gap += 1
    if not demand: return np.zeros(h)
    z, p = demand[0], intervals[0]
    for d, q in zip(demand[1:], intervals[1:]):
        z += alpha * (d - z)
        p += alpha * (q - p)
    sba = (1 - alpha/2) * (z / max(p, 1e-6))
    return np.full(h, sba)

@app.get("/health")
def health(): return {"status": "ok", "version": APP_VERSION}

@app.post("/jobs/run")
def run_job(req: RunRequest):
    global FREQUENCY
    h = req.horizon or HORIZON
    freq = (req.frequency or FREQUENCY).upper()
    FREQUENCY = freq
    df = load_base()
    results = {"horizon": h, "frequency": freq, "version": APP_VERSION, "generated_at": datetime.utcnow().isoformat()}
    registry, fc_list = [], []
    for cat, g in df.groupby("categoria"):
        y = g.sort_values("ds").set_index("ds")["y"].asfreq(freq, fill_value=0)
        model_name = CATEGORIA_TO_MODEL.get(cat.lower(), "ETS")
        try:
            if model_name == "ETS":
                fit = fit_ets(y)
                mean = fit.forecast(h)
                conf = None
                reg = {"categoria": cat, "modello": "ETS", "params": {"alpha": 0.4, "beta": 0.2, "gamma": 0.2, "damped_trend": True}}
            elif model_name == "SARIMAX":
                fit = fit_sarimax(y)
                fc = fit.get_forecast(steps=h)
                mean = fc.predicted_mean
                conf = fc.conf_int(alpha=0.2)
                reg = {"categoria": cat, "modello": "SARIMAX", "params": {"order": [0,1,1], "seasonal_order": [1,1,0, SEASONAL_PERIODS], "trend": "c"}}
            elif model_name == "PROPHET" and _HAS_PROPHET:
                dfp = y.reset_index().rename(columns={"index": "ds", "y": "y"})
                m = fit_prophet(dfp)
                future = m.make_future_dataframe(periods=h, freq=freq)
                pred = m.predict(future).set_index("ds")
                mean = pred["yhat"].tail(h)
                conf = None
                reg = {"categoria": cat, "modello": "Prophet", "params": {"changepoint_prior_scale": 0.10}}
            elif model_name == "CROSTON_SBA":
                mean_vals = croston_sba_forecast(y, h)
                idx = pd.date_range(y.index.max() + pd.tseries.frequencies.to_offset(freq), periods=h, freq=freq)
                mean = pd.Series(mean_vals, index=idx)
                conf = None
                reg = {"categoria": cat, "modello": "Croston_SBA", "params": {"alpha": 0.2}}
            else:
                fit = fit_ets(y)
                mean = fit.forecast(h)
                conf = None
                reg = {"categoria": cat, "modello": "ETS", "params": {"seasonal_periods": SEASONAL_PERIODS}}

            for i, ds in enumerate(mean.index):
                fc_list.append({
                    "categoria": cat,
                    "ds": ds.date().isoformat(),
                    "yhat": float(max(0.0, mean.iloc[i])),
                    "yhat_lower": float(max(0.0, conf.iloc[i,0])) if conf is not None else None,
                    "yhat_upper": float(max(0.0, conf.iloc[i,1])) if conf is not None else None,
                })
            reg["timestamp"] = datetime.utcnow().isoformat()
            registry.append(reg)
        except Exception as e:
            registry.append({"categoria": cat, "modello": "ERROR", "error": str(e), "timestamp": datetime.utcnow().isoformat()})

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(ARTIFACTS_DIR, f"forecasts_{ts}.json"), "w") as f:
        json.dump(fc_list, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, f"model_registry_{ts}.json"), "w") as f:
        json.dump(registry, f, indent=2)

    results["summary"] = {c["categoria"]: c["modello"] for c in registry}
    return results

@app.get("/forecasts/latest", response_model=List[ForecastItem])
def forecasts_latest():
    files = [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("forecasts_")]
    if not files:
        raise HTTPException(status_code=404, detail="Nessun forecast generato")
    files.sort()
    with open(os.path.join(ARTIFACTS_DIR, files[-1])) as f:
        return json.load(f)

@app.get("/model-registry")
def model_registry():
    files = [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith("model_registry_")]
    if not files:
        raise HTTPException(status_code=404, detail="Nessun registry generato")
    files.sort()
    with open(os.path.join(ARTIFACTS_DIR, files[-1])) as f:
        return json.load(f)
