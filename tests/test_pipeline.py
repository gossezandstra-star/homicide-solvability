"""Fast unit tests. They use a tiny synthetic dataset, so they run without the Kaggle file.

Run with ``pytest -q`` from the project root.
"""

import numpy as np
import pandas as pd
import pytest

from homicide.cluster import choose_k, summarise
from homicide.config import load_config
from homicide.data import clean, time_split
from homicide.features import build_preprocessor, input_columns
from homicide.train import best_threshold, param_grid, scores


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame({
        "Record ID": np.arange(n), "Agency Code": rng.choice(["A1", "A2", "A3"], n),
        "Agency Type": rng.choice(["Municipal Police", "Sheriff"], n),
        "City": rng.choice(["Lake", "Marion"], n), "State": "Indiana",
        "Year": rng.integers(1980, 2015, n), "Month": rng.choice(["January", "July"], n),
        "Crime Type": "Murder or Manslaughter", "Crime Solved": rng.choice(["Yes", "No"], n),
        "Victim Sex": rng.choice(["Male", "Female"], n),
        "Victim Age": rng.choice([0, 25, 40, 998], n),
        "Victim Race": rng.choice(["White", "Black"], n),
        "Victim Ethnicity": "Unknown", "Weapon": rng.choice(["Handgun", "Knife"], n),
        "Victim Count": 0, "Record Source": "FBI",
        "Perpetrator Age": "0", "Relationship": "Unknown", "Perpetrator Count": 0,
    })


def test_clean_drops_leakage_and_codes_unknown_age(raw, cfg):
    df = clean(raw, cfg["features"]["leakage"])
    assert not set(cfg["features"]["leakage"]) & set(df.columns)
    assert df.loc[raw["Victim Age"] == 998, "Victim Age"].isna().all()
    assert set(df["y"]) <= {0, 1}
    assert (df["y"] == (raw["Crime Solved"] == "No")).all()


def test_time_split_has_no_overlap(raw, cfg):
    df = clean(raw, cfg["features"]["leakage"])
    tr, va, te = time_split(df, 2008, 2011)
    assert tr["Year"].max() <= 2008 < va["Year"].min()
    assert va["Year"].max() <= 2011 < te["Year"].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_preprocessor_handles_unseen_categories(raw, cfg):
    df = clean(raw, cfg["features"]["leakage"])
    pre = build_preprocessor(cfg).fit(df[input_columns(cfg)], df["y"])
    new = df[input_columns(cfg)].head(3).copy()
    new["Weapon"] = "Laser"            # never seen during training
    new["Agency Code"] = "ZZ999"
    assert np.isfinite(pre.transform(new)).all()


def test_param_grid_and_metrics():
    assert len(param_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    s = scores(y, p, 0.5)
    assert s["precision"] == 1.0 and s["recall"] == 0.5
    assert 0 < best_threshold(y, p) <= 0.8


def test_choose_k_picks_the_elbow():
    curve = pd.DataFrame({"k": [10, 20, 30, 40, 50], "cost": [100, 40, 30, 25, 22]})
    assert choose_k(curve) == 20


def test_cluster_summary_flags_high_unsolved_cluster():
    df = pd.DataFrame({"y": [1] * 30 + [0] * 70 + [1] * 10 + [0] * 90, "Year": 2000,
                       "City": ["Lake"] * 100 + ["Marion"] * 100})
    labels = np.array([0] * 30 + [1] * 70 + [1] * 100)
    table = summarise(df, labels, ["City"], min_size=15, alpha=0.01)
    assert table.loc[table["cluster"] == 0, "flagged"].item()
    assert not table.loc[table["cluster"] == 1, "flagged"].item()


def test_predict_handles_missing_columns(raw, cfg, tmp_path):
    """A CSV with only a few columns (no victim age, no agency) can still be scored."""
    import joblib  # pylint: disable=import-outside-toplevel
    from sklearn.tree import DecisionTreeClassifier  # pylint: disable=import-outside-toplevel
    from homicide.predict import SolvabilityModel  # pylint: disable=import-outside-toplevel
    from homicide.train import make_pipeline  # pylint: disable=import-outside-toplevel

    df = clean(raw, cfg["features"]["leakage"])
    pipe = make_pipeline(cfg, DecisionTreeClassifier(max_depth=3, random_state=0))
    pipe.fit(df[input_columns(cfg)], df["y"])
    path = tmp_path / "m.joblib"
    joblib.dump({"model_name": "t", "pipeline": pipe, "threshold": 0.5,
                 "input_columns": input_columns(cfg)}, path)
    out = SolvabilityModel(path).predict(pd.DataFrame({"Year": [2014], "State": ["Texas"],
                                                       "Weapon": ["Handgun"]}))
    assert 0 <= out["p_unsolved"].iloc[0] <= 1
