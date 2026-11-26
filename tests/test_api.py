"""API tests. They train a tiny model on synthetic data, so they run without the Kaggle
file and without the real model (also in CI)."""

import importlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from homicide.config import load_config
from homicide.data import clean
from homicide.features import input_columns
from homicide.train import make_pipeline


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    """A TestClient for the app, backed by a small model trained on random cases."""
    from fastapi.testclient import TestClient  # pylint: disable=import-outside-toplevel

    cfg = load_config()
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame({
        "Agency Code": rng.choice(["A1", "A2"], n), "Agency Type": "Municipal Police",
        "City": rng.choice(["Lake", "Marion"], n), "State": rng.choice(["Indiana", "Texas"], n),
        "Year": rng.integers(1980, 2009, n), "Month": "July",
        "Crime Type": "Murder or Manslaughter", "Crime Solved": rng.choice(["Yes", "No"], n),
        "Victim Sex": rng.choice(["Male", "Female"], n), "Victim Age": rng.integers(0, 90, n),
        "Victim Race": rng.choice(["White", "Black"], n), "Victim Ethnicity": "Unknown",
        "Weapon": rng.choice(["Handgun", "Strangulation"], n), "Victim Count": 0,
        "Record Source": "FBI"})
    df = clean(raw)
    pipe = make_pipeline(cfg, RandomForestClassifier(n_estimators=10, random_state=0))
    pipe.fit(df[input_columns(cfg)], df["y"])
    path = tmp_path_factory.mktemp("model") / "best_model.joblib"
    joblib.dump({"model_name": "test_forest", "pipeline": pipe, "threshold": 0.5,
                 "input_columns": input_columns(cfg)}, path)

    monkeypatch_module.setenv("MODEL_PATH", str(path))
    monkeypatch_module.setenv("API_KEY", "secret")
    monkeypatch_module.setenv("MAX_CASES", "5")
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    yield TestClient(main.app)
    sys.modules.pop("app.main", None)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


CASES = [{"Year": 2014, "State": "Indiana", "City": "Lake", "Victim Sex": "Female",
          "Victim Age": 31, "Victim Race": "Black", "Weapon": "Strangulation"},
         {"Year": 2014, "State": "Texas", "Weapon": "Handgun"}]
KEY = {"X-API-Key": "secret"}


def test_health_is_open(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_requires_key(client):
    assert client.post("/predict", json=CASES).status_code == 401
    assert client.post("/predict", json=CASES, headers={"X-API-Key": "wrong"}).status_code == 401


def test_predict_and_metrics(client):
    r = client.post("/predict", json=CASES, headers=KEY)
    assert r.status_code == 200
    out = r.json()
    assert len(out) == 2 and all(0 <= o["p_unsolved"] <= 1 for o in out)
    m = client.get("/metrics", headers=KEY).json()
    assert m["traffic"]["cases_scored"] == 2
    assert set(m) >= {"latency_ms", "traffic", "errors", "saturation"}


def test_input_validation(client):
    assert client.post("/predict", json=CASES * 3, headers=KEY).status_code == 422   # > MAX_CASES
    bad = [{"Year": 2014, "State": "Texas", "Victim Age": -5}]
    assert client.post("/predict", json=bad, headers=KEY).status_code == 422
    assert client.post("/predict", json=[], headers=KEY).status_code == 422
