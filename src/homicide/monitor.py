"""Drift and performance monitoring (lecture 17, Monitoring & Dashboarding).

In production the label ("was the case solved?") arrives months or years after the
prediction (a *feedback delay*), so we cannot rely on accuracy alone. We therefore
monitor three things, using the training years as the reference and the test years
(2012-2014) as a stand-in for production data:

1. **Data drift** per input feature
   * numeric features: two-sample Kolmogorov-Smirnov test;
   * categorical features: Pearson chi-square test on the category counts (agency and
     county: their 100 most frequent values plus "other");
   * all features: Population Stability Index (PSI). Rule of thumb: < 0.1 stable,
     0.1-0.25 moderate shift, > 0.25 significant shift (Yurdakul, 2018).
2. **Prediction drift**: PSI of the predicted P(unsolved), validation vs test years.
3. **Performance per year** on the test years, once labels are available.

Outputs (in ``reports/metrics``): ``drift_report.csv`` and ``performance_by_year.csv``.

Run::

    python -m homicide.monitor
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from homicide.config import load_config
from homicide.data import load_splits
from homicide.train import scores

log = logging.getLogger(__name__)
EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two discrete distributions (proportions)."""
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_drift(ref: pd.Series, cur: pd.Series, bins: int) -> dict:
    """KS test and PSI for a numeric feature.

    PSI uses quantile bins of the reference. A feature with only a few distinct values
    (such as the number of additional victims) gets one bin per value instead, because
    quantile edges would collapse into a single bin and hide any shift.
    """
    ref, cur = ref.dropna(), cur.dropna()
    if ref.nunique() <= bins:
        values = np.sort(ref.unique())
        edges = np.concatenate([[-np.inf], (values[:-1] + values[1:]) / 2, [np.inf]])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(cur, edges)[0] / len(cur)
    ks = ks_2samp(ref, cur)
    return {"test": "Kolmogorov-Smirnov", "statistic": ks.statistic, "p_value": ks.pvalue,
            "psi": psi(e, a)}


def categorical_drift(ref: pd.Series, cur: pd.Series, top: int | None = None) -> dict:
    """Chi-square test and PSI for a categorical feature.

    For high-cardinality columns (agency, county) only the ``top`` most frequent reference
    categories are kept and the rest is pooled as "other", so that tiny categories do not
    dominate the statistics.
    """
    ref, cur = ref.astype(str), cur.astype(str)
    if top is not None:
        keep_cats = set(ref.value_counts().index[:top])
        ref = ref.where(ref.isin(keep_cats), "other")
        cur = cur.where(cur.isin(keep_cats), "other")
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts().reindex(cats, fill_value=0)
    c = cur.value_counts().reindex(cats, fill_value=0)
    keep = (r + c) > 0
    chi2, p, _, _ = chi2_contingency(np.vstack([r[keep], c[keep]]))
    return {"test": "Pearson chi-square", "statistic": chi2, "p_value": p,
            "psi": psi(r / r.sum(), c / c.sum())}


def status(value: float, warn: float, alert: float) -> str:
    """Turn a PSI value into a readable status."""
    return "significant shift" if value > alert else "moderate shift" if value > warn else "stable"


def run(cfg: dict) -> pd.DataFrame:
    """Run this pipeline step with the given config."""
    m = cfg["monitoring"]
    out = cfg["paths"]["metrics_dir"]
    train, val, test = load_splits(cfg)
    art = joblib.load(cfg["paths"]["models_dir"] / "best_model.joblib")
    cols = art["input_columns"]

    rows = []
    for col in cfg["features"]["numeric"]:
        if col == "Year":          # differs by construction of the time split
            continue
        rows.append({"feature": col, **numeric_drift(train[col], test[col], m["psi_bins"])})
    for col in cfg["features"]["categorical"]:
        rows.append({"feature": col, **categorical_drift(train[col], test[col])})
    for col in cfg["features"]["target_encoded"]:       # agency and county: top 100 + other
        rows.append({"feature": col, **categorical_drift(train[col], test[col], top=100)})

    # Prediction drift: validation years (just before deployment) vs test years
    p_val = art["pipeline"].predict_proba(val[cols])[:, 1]
    p_test = art["pipeline"].predict_proba(test[cols])[:, 1]
    rows.append({"feature": "prediction P(unsolved)",
                 **numeric_drift(pd.Series(p_val), pd.Series(p_test), m["psi_bins"])})
    # Target drift (only measurable once labels arrive)
    rows.append({"feature": "label: unsolved share",
                 **categorical_drift(train["y"], test["y"])})

    report = pd.DataFrame(rows)
    report["status"] = report["psi"].apply(status, args=(m["psi_warning"], m["psi_alert"]))
    report.sort_values("psi", ascending=False).to_csv(out / "drift_report.csv", index=False)
    log.info("\n%s", report.round(4).to_string(index=False))

    # Performance per year on the test years
    perf = []
    for year, g in test.assign(p=p_test).groupby("Year"):
        s = scores(g["y"].to_numpy(), g["p"].to_numpy(), art["threshold"])
        perf.append({"year": int(year), "n": len(g), "unsolved_share": g["y"].mean(), **s})
    pd.DataFrame(perf).to_csv(out / "performance_by_year.csv", index=False)
    return report


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
