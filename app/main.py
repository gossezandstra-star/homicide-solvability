"""Lightweight REST API around the solvability model (FastAPI, lecture 14).

Start locally::

    uvicorn app.main:app --port 8000

or with Docker / Kubernetes (see README). Interactive docs: http://localhost:8000/docs

Endpoints
---------
GET  /health    -- liveness check for uptime monitoring and the Kubernetes probes
GET  /metrics   -- the four golden signals (latency, traffic, errors, saturation) plus
                   the share of flagged cases, to spot prediction drift (lecture 17)
POST /predict   -- one or more cases in, probabilities and flags out
"""

from __future__ import annotations

import os
import resource
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from homicide.predict import SolvabilityModel

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parents[1]
                                 / "models" / "best_model.joblib"))

app = FastAPI(title="Homicide solvability API", version="1.0.0")
model = SolvabilityModel(MODEL_PATH)

STARTED = time.time()
STATS = {"requests": 0, "errors": 0, "cases_scored": 0, "cases_flagged": 0}
LATENCIES_MS: deque[float] = deque(maxlen=1000)     # last 1000 requests
P_UNSOLVED: deque[float] = deque(maxlen=5000)       # last 5000 predictions


class Case(BaseModel):
    """Information that is known when a homicide case is opened."""
    model_config = ConfigDict(populate_by_name=True)

    year: int = Field(alias="Year", examples=[2014])
    month: str = Field("Unknown", alias="Month", examples=["July"])
    state: str = Field(alias="State", examples=["Indiana"])
    city: str = Field("Unknown", alias="City", examples=["Lake"])
    agency_code: str = Field("Unknown", alias="Agency Code", examples=["IN04500"])
    agency_type: str = Field("Unknown", alias="Agency Type", examples=["Municipal Police"])
    crime_type: str = Field("Murder or Manslaughter", alias="Crime Type")
    victim_sex: str = Field("Unknown", alias="Victim Sex", examples=["Female"])
    victim_age: float | None = Field(None, alias="Victim Age", examples=[31])
    victim_race: str = Field("Unknown", alias="Victim Race", examples=["Black"])
    victim_ethnicity: str = Field("Unknown", alias="Victim Ethnicity")
    weapon: str = Field("Unknown", alias="Weapon", examples=["Strangulation"])
    victim_count: int = Field(0, alias="Victim Count", description="additional victims")
    record_source: str = Field("FBI", alias="Record Source")


class Prediction(BaseModel):
    p_unsolved: float
    p_solved: float
    likely_to_go_cold: bool


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Record latency, traffic and errors for every request."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        STATS["errors"] += 1
        raise
    if request.url.path not in ("/metrics", "/health"):
        STATS["requests"] += 1
        LATENCIES_MS.append((time.perf_counter() - start) * 1000)
        if response.status_code >= 400:
            STATS["errors"] += 1
    return response


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "model": model.name, "threshold": model.threshold}


@app.get("/metrics")
def metrics() -> dict:
    """Four golden signals plus prediction statistics."""
    uptime = time.time() - STARTED
    lat = np.array(LATENCIES_MS) if LATENCIES_MS else np.array([0.0])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                       "max": float(lat.max())},
        "traffic": {"requests": STATS["requests"], "cases_scored": STATS["cases_scored"],
                    "requests_per_minute": STATS["requests"] / max(uptime / 60, 1e-9)},
        "errors": {"count": STATS["errors"],
                   "rate": STATS["errors"] / max(STATS["requests"], 1)},
        "saturation": {"max_memory_mb": usage.ru_maxrss / 1024,
                       "cpu_seconds": usage.ru_utime + usage.ru_stime},
        "predictions": {"mean_p_unsolved": float(np.mean(P_UNSOLVED)) if P_UNSOLVED else None,
                        "flag_rate": STATS["cases_flagged"] / max(STATS["cases_scored"], 1)},
        "uptime_seconds": uptime,
    }


@app.post("/predict", response_model=list[Prediction])
def predict(cases: list[Case]) -> list[Prediction]:
    """Score one or more cases."""
    df = pd.DataFrame([c.model_dump(by_alias=True) for c in cases])
    df["Victim Age"] = df["Victim Age"].fillna(998)       # 998 = unknown in the SHR coding
    result = model.predict(df)
    STATS["cases_scored"] += len(result)
    STATS["cases_flagged"] += int(result["likely_to_go_cold"].sum())
    P_UNSOLVED.extend(result["p_unsolved"].tolist())
    return [Prediction(p_unsolved=r.p_unsolved, p_solved=r.p_solved,
                       likely_to_go_cold=bool(r.likely_to_go_cold))
            for r in result.itertuples()]
