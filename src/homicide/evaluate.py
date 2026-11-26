"""Evaluate the trained models on the held-out test years (2012-2014).

The *selected* model (chosen on validation in ``train.py``) is the one we report as our
result. For transparency the other models are evaluated on the test set too, with the
threshold each of them got on validation; they were not re-tuned on test.

Also produced:

* a bias audit: error rates of the selected model per victim race, sex, ethnicity and
  age group, and the effect of per-group thresholds that equalise the false-positive
  rate across victim races (chosen on validation);
* paired bootstrap 95% intervals of every model's test average precision and of its
  difference with the selected model;
* a calibration table (predicted vs observed unsolved share per decile, Brier score);
* permutation feature importance of the selected model (on the validation set);
* examples of the most confident wrong predictions, for the error analysis.

Outputs (in ``reports/metrics``): ``test_results.csv``, ``test_predictions.parquet``,
``bootstrap_ci.csv``, ``bias_audit.csv``, ``fairness_thresholds.csv``, ``group_thresholds.json``,
``calibration.csv``, ``calibration_summary.json``, ``feature_importance.csv``,
``error_examples.csv``.

Run::

    python -m homicide.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score)

from homicide.config import load_config, set_seed
from homicide.data import load_splits
from homicide.train import scores

log = logging.getLogger(__name__)

AUDIT_GROUPS = ["Victim Race", "Victim Sex", "Victim Ethnicity", "Victim Age Group"]
FAIRNESS_GROUP = "Victim Race"
ERROR_COLUMNS = ["Year", "State", "City", "Agency Type", "Victim Sex", "Victim Age",
                 "Victim Race", "Weapon", "Crime Solved"]


def calibration(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table: mean predicted vs observed unsolved share per probability decile."""
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "p": proba, "y": y})
    return df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                 observed=("y", "mean")).reset_index()


def group_thresholds(val: pd.DataFrame, p_val: np.ndarray, base_thr: float,
                     group: str, min_n: int = 300) -> dict:
    """Per-group thresholds, chosen on validation, that give every group the same
    false-positive rate as the whole validation set has at the base threshold."""
    y = val["y"].to_numpy()
    target_fpr = float(((p_val >= base_thr) & (y == 0)).sum() / (y == 0).sum())
    thresholds = {}
    for value, g_idx in val.reset_index(drop=True).groupby(group).groups.items():
        neg = p_val[np.asarray(g_idx)][y[np.asarray(g_idx)] == 0]
        if len(g_idx) < min_n or len(neg) == 0:
            thresholds[value] = base_thr
            continue
        thresholds[value] = float(np.quantile(neg, 1 - target_fpr))   # FPR = target
    return {"target_fpr": target_fpr, "thresholds": thresholds}


def fairness_comparison(test: pd.DataFrame, p_test: np.ndarray, base_thr: float,
                        group: str, per_group: dict) -> pd.DataFrame:
    """Error rates per group with one threshold vs with per-group thresholds."""
    t = test.reset_index(drop=True)
    y = t["y"].to_numpy()
    thr_vec = t[group].map(per_group).fillna(base_thr).to_numpy()
    rows = []
    for label, pred in (("single threshold", p_test >= base_thr),
                        ("per-group thresholds", p_test >= thr_vec)):
        pred = pred.astype(int)
        rows.append({"policy": label, "group": "all", "n": len(y),
                     "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                     "f1": f1_score(y, pred),
                     "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / (y == 0).sum())})
        for value, g in t.groupby(group):
            m = t.index.isin(g.index)
            if m.sum() < 300:
                continue
            yy, pp = y[m], pred[m]
            rows.append({"policy": label, "group": value, "n": int(m.sum()),
                         "precision": precision_score(yy, pp, zero_division=0),
                         "recall": recall_score(yy, pp, zero_division=0),
                         "f1": f1_score(yy, pp, zero_division=0),
                         "false_positive_rate": float(((pp == 1) & (yy == 0)).sum() / (yy == 0).sum())})
    return pd.DataFrame(rows)


def bootstrap_comparison(preds: pd.DataFrame, reference: str, n_boot: int = 1000,
                         seed: int = 42) -> pd.DataFrame:
    """95% bootstrap intervals for each model's test average precision and for its
    difference with the reference (selected) model. Cases are resampled with
    replacement; every model is scored on the same resample (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    y = preds["y"].to_numpy()
    models = [c[2:] for c in preds.columns if c.startswith("p_")]
    probas = {m: preds[f"p_{m}"].to_numpy() for m in models}
    draws = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for m in models:
            draws[m][b] = average_precision_score(y[idx], probas[m][idx])
    rows = []
    for m in models:
        diff = draws[m] - draws[reference]
        rows.append({"model": m, "average_precision": average_precision_score(y, probas[m]),
                     "ap_ci_low": np.percentile(draws[m], 2.5),
                     "ap_ci_high": np.percentile(draws[m], 97.5),
                     f"diff_vs_{reference}": float(np.mean(diff)),
                     "diff_ci_low": np.percentile(diff, 2.5),
                     "diff_ci_high": np.percentile(diff, 97.5)})
    return pd.DataFrame(rows)


def bias_audit(df: pd.DataFrame, proba: np.ndarray, threshold: float,
               groups: list[str]) -> pd.DataFrame:
    """Per-group error rates. Positive class = unsolved."""
    pred = (proba >= threshold).astype(int)
    rows = []
    for col in groups:
        for value, idx in df.groupby(col).groups.items():
            mask = df.index.isin(idx)
            y, p = df["y"].to_numpy()[mask], pred[mask]
            if len(y) < 100:      # too small to say anything reliable
                continue
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            rows.append({
                "attribute": col, "group": value, "n": int(len(y)),
                "unsolved_share": float(y.mean()),
                "flag_rate": float(p.mean()),                        # share flagged "likely cold"
                "recall": tp / (tp + fn) if tp + fn else np.nan,     # unsolved cases we catch
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
            })
    return pd.DataFrame(rows)


def run(cfg: dict) -> dict:
    """Run this pipeline step with the given config."""
    set_seed(cfg["seed"])
    models_dir, out = cfg["paths"]["models_dir"], cfg["paths"]["metrics_dir"]
    _, val, test = load_splits(cfg)
    with open(out / "validation_summary.json", encoding="utf-8") as f:
        val_summary = json.load(f)
    best = joblib.load(models_dir / "best_model.joblib")
    cols, y_test = best["input_columns"], test["y"].to_numpy()

    # Test metrics for every model (baseline = always "solved", P(unsolved) = 0).
    rows = [{"model": "majority_baseline", "threshold": 0.5,
             **scores(y_test, np.zeros(len(y_test)))}]
    probas = {}
    for name in cfg["models"]:
        pipe = joblib.load(models_dir / f"{name}.joblib")
        thr = val_summary[name]["threshold"]
        probas[name] = pipe.predict_proba(test[cols])[:, 1]
        rows.append({"model": name, "threshold": thr, **scores(y_test, probas[name], thr)})
    results = pd.DataFrame(rows)
    results["selected"] = results["model"] == best["model_name"]
    results.to_csv(out / "test_results.csv", index=False)
    log.info("\n%s", results.round(4).to_string(index=False))

    # Predictions of all models, used by figures.py
    preds = test[["Record ID", "y", *AUDIT_GROUPS]].copy()
    for name, p in probas.items():
        preds[f"p_{name}"] = p
    preds.to_parquet(out / "test_predictions.parquet", index=False)
    bootstrap_comparison(preds, best["model_name"], seed=cfg["seed"]).to_csv(
        out / "bootstrap_ci.csv", index=False)

    # Bias audit of the selected model
    p_best, thr = probas[best["model_name"]], best["threshold"]
    audit = bias_audit(test.reset_index(drop=True), p_best, thr, AUDIT_GROUPS)
    audit.to_csv(out / "bias_audit.csv", index=False)

    # Calibration of the selected model
    calib = calibration(y_test, p_best)
    calib.to_csv(out / "calibration.csv", index=False)
    with open(out / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump({"brier_score": brier_score_loss(y_test, p_best),
                   "brier_score_baseline": brier_score_loss(y_test, np.full(len(y_test),
                                                                            y_test.mean()))},
                  f, indent=2)

    # Fairer thresholds: equal false-positive rate per victim race, chosen on validation
    p_val = best["pipeline"].predict_proba(val[cols])[:, 1]
    gt = group_thresholds(val, p_val, thr, FAIRNESS_GROUP)
    fairness_comparison(test, p_best, thr, FAIRNESS_GROUP, gt["thresholds"]).to_csv(
        out / "fairness_thresholds.csv", index=False)
    with open(out / "group_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

    # Most confident mistakes
    t = test.reset_index(drop=True).assign(p_unsolved=p_best)
    fn = t[(t["y"] == 1)].nsmallest(10, "p_unsolved").assign(error="missed unsolved case")
    fp = t[(t["y"] == 0)].nlargest(10, "p_unsolved").assign(error="false alarm")
    pd.concat([fn, fp])[["error", "p_unsolved", *ERROR_COLUMNS]].to_csv(
        out / "error_examples.csv", index=False)

    # Permutation importance on (a sample of) the validation set, on raw input columns
    sample = val.sample(n=min(15000, len(val)), random_state=cfg["seed"])
    imp = permutation_importance(best["pipeline"], sample[cols], sample["y"],
                                 scoring="average_precision", n_repeats=5,
                                 random_state=cfg["seed"], n_jobs=1)
    importance = pd.DataFrame({"feature": cols, "importance_mean": imp.importances_mean,
                               "importance_std": imp.importances_std})
    importance.sort_values("importance_mean", ascending=False).to_csv(
        out / "feature_importance.csv", index=False)

    selected = results[results["selected"]].iloc[0].to_dict()
    log.info("selected model on test: %s", selected)
    return selected


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
