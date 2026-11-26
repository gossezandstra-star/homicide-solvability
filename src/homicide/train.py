"""Train the solvability classifiers and select the best one on the validation set.

Steps
-----
1. Majority-class baseline (always predicts "solved").
2. Models from the course: logistic regression (lecture 4), decision tree with the gini
   and entropy criteria, random forest (bagging) and AdaBoost (boosting) (lecture 9),
   plus histogram gradient boosting as a stronger boosting variant.
   Every hyper-parameter combination from ``config.yaml`` is fitted on a stratified
   sample of the training years and scored on the validation years (hold-out
   validation; a time-based split instead of k-fold, because the model will be used on
   future years).
3. The best setting of each model is refitted on the full training set.
4. The model with the highest validation score is kept, together with the decision
   threshold that maximises F1 for the "unsolved" class on validation.

Outputs: ``models/best_model.joblib``, ``reports/metrics/tuning_results.csv`` and
``reports/metrics/validation_summary.json``.

Run::

    python -m homicide.train
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from homicide.config import load_config, set_seed
from homicide.data import load_splits
from homicide.features import build_preprocessor, input_columns

log = logging.getLogger(__name__)


def make_estimator(name: str, params: dict, seed: int):
    """Return an unfitted classifier for a model name and a parameter setting."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, **params)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    if name == "adaboost":
        params = dict(params)
        depth = params.pop("base_max_depth")
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth),
                                  random_state=seed, **params)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=True,
                                              validation_fraction=0.1, **params)
    raise ValueError(f"unknown model {name}")


def make_pipeline(cfg: dict, estimator) -> Pipeline:
    """Preprocessing + classifier in one scikit-learn Pipeline."""
    return Pipeline([("prep", build_preprocessor(cfg)), ("model", estimator)])


def param_grid(grid: dict) -> list[dict]:
    """Expand {"a": [1, 2], "b": [3]} into [{"a": 1, "b": 3}, {"a": 2, "b": 3}]."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def scores(y_true, proba, threshold: float = 0.5) -> dict:
    """Classification metrics with "unsolved" (1) as the positive class."""
    pred = (proba >= threshold).astype(int)
    return {
        "average_precision": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
    }


def best_threshold(y_true, proba) -> float:
    """Threshold on P(unsolved) that maximises F1 for the unsolved class."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def run(cfg: dict) -> dict:
    """Run this pipeline step with the given config."""
    seed = cfg["seed"]
    set_seed(seed)
    train, val, _ = load_splits(cfg)
    cols = input_columns(cfg)
    X_train, y_train = train[cols], train["y"].to_numpy()
    X_val, y_val = val[cols], val["y"].to_numpy()

    # Stratified sample of the training years for fast hyper-parameter search.
    n = min(cfg["tuning"]["sample_size"], len(train))
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=n,
                                            stratify=y_train, random_state=seed)
    metric = cfg["selection_metric"]
    rows, best_per_model = [], {}

    # 1. Baseline
    baseline = make_pipeline(cfg, DummyClassifier(strategy="most_frequent")).fit(X_train, y_train)
    base_scores = scores(y_val, baseline.predict_proba(X_val)[:, 1])
    rows.append({"model": "majority_baseline", "params": "{}", **base_scores, "fit_seconds": 0.0})
    log.info("baseline: %s", {k: round(v, 3) for k, v in base_scores.items()})

    # 2. Grid search on the validation set
    for name, grid in cfg["models"].items():
        best_params, best_score = None, -np.inf
        for params in param_grid(grid):
            t0 = time.time()
            pipe = make_pipeline(cfg, make_estimator(name, params, seed)).fit(X_tune, y_tune)
            s = scores(y_val, pipe.predict_proba(X_val)[:, 1])
            secs = time.time() - t0
            rows.append({"model": name, "params": json.dumps(params), **s, "fit_seconds": secs})
            log.info("%-22s %-70s %s=%.4f (%.0fs)", name, params, metric, s[metric], secs)
            if s[metric] > best_score:
                best_params, best_score = params, s[metric]
        best_per_model[name] = best_params

    tuning = pd.DataFrame(rows)
    tuning.to_csv(cfg["paths"]["metrics_dir"] / "tuning_results.csv", index=False)

    # 3. Refit the best setting of every model on the full training set
    summary, fitted = {"majority_baseline": {"params": {}, **base_scores}}, {}
    for name, params in best_per_model.items():
        t0 = time.time()
        pipe = make_pipeline(cfg, make_estimator(name, params, seed)).fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        thr = best_threshold(y_val, proba)
        summary[name] = {"params": params, "threshold": thr,
                         **{f"{k}": v for k, v in scores(y_val, proba, thr).items()},
                         "refit_seconds": time.time() - t0}
        fitted[name] = pipe
        joblib.dump(pipe, cfg["paths"]["models_dir"] / f"{name}.joblib", compress=3)
        log.info("refit %-22s val %s=%.4f, F1(unsolved)=%.4f at threshold %.3f",
                 name, metric, summary[name][metric], summary[name]["f1"], thr)

    # 4. Select the overall best model
    best_name = max(best_per_model, key=lambda m: summary[m][metric])
    artifact = {
        "model_name": best_name,
        "pipeline": fitted[best_name],
        "threshold": summary[best_name]["threshold"],
        "input_columns": cols,
        "positive_class": "unsolved",
        "train_years": [int(train["Year"].min()), int(train["Year"].max())],
    }
    joblib.dump(artifact, cfg["paths"]["models_dir"] / "best_model.joblib", compress=3)
    summary["selected_model"] = best_name
    with open(cfg["paths"]["metrics_dir"] / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    log.info("selected model: %s", best_name)
    return summary


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
