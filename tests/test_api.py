"""API smoke test. Skipped when no trained model is present (e.g. in CI)."""

from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists() or MODEL.stat().st_size == 0,
                                reason="no trained model")


def test_predict_and_metrics():
    from fastapi.testclient import TestClient  # pylint: disable=import-outside-toplevel
    from app.main import app                    # pylint: disable=import-outside-toplevel

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    cases = [{"Year": 2014, "State": "Indiana", "City": "Lake", "Victim Sex": "Female",
              "Victim Age": 31, "Victim Race": "Black", "Weapon": "Strangulation"},
             {"Year": 2014, "State": "Texas", "Weapon": "Handgun"}]
    r = client.post("/predict", json=cases)
    assert r.status_code == 200
    out = r.json()
    assert len(out) == 2 and all(0 <= o["p_unsolved"] <= 1 for o in out)
    m = client.get("/metrics").json()
    assert m["traffic"]["cases_scored"] == 2 and m["errors"]["count"] == 0
