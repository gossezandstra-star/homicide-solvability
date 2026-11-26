"""Evaluate the trained models on the held-out test years (2012-2014).

The *selected* model (chosen on validation in ``train.py``) is the one we report as our
result. For transparency the other models are evaluated on the test set too, with the
threshold each of them got on validation; they were not re-tuned on test.

Also produced:

* a bias audit: error rates of the selected model per victim race and victim sex;
* permutation feature importance of the selected model (on the validation set);
* examples of the most confident wrong predictions, for the error analysis.

Outputs (in ``reports/metrics``): ``test_results.csv``, ``test_predictions.parquet``,
``bias_audit.csv``, ``feature_importance.csv``, ``error_examples.csv``.

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
from sklearn.metrics import confusion_matrix

from homicide.config import load_config, set_seed
from homicide.data import load_splits
from homicide.train import scores

log = logging.getLogger(__name__)

AUDIT_GROUPS = ["Victim Race", "Victim Sex"]
ERROR_COLUMNS = ["Year", "State", "City", "Agency Type", "Victim Sex", "Victim Age",
                 "Victim Race", "Weapon", "Crime Solved"]


def bias_audit(df: pd.DataFrame, proba: np.ndarray, threshold: float, groups: list[str]) -> pd.DataFrame:
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

    # Bias audit of the selected model
    p_best, thr = probas[best["model_name"]], best["threshold"]
    audit = bias_audit(test.reset_index(drop=True), p_best, thr, AUDIT_GROUPS)
    audit.to_csv(out / "bias_audit.csv", index=False)

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
    log.info("selected model on test: %s", {k: v for k, v in selected.items()})
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
