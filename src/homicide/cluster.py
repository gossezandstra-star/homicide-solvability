"""Cluster detector for possible serial-offender patterns.

Idea (inspired by the Murder Accountability Project): group all homicides in a region
that share the same kind of victim, place and weapon. A group that is large and has a
significantly higher share of unsolved cases than the rest of the region is worth a
second look by investigators.

Because every clustering feature is categorical, we use **k-modes** (Huang, 1998):
k-means (lecture 7) with the Hamming distance and cluster *modes* instead of means, much
like k-medoids uses real data points as centres. The number of clusters is chosen with
the elbow method on the k-modes cost (lecture 7); the silhouette score (Rousseeuw, 1987)
with the Hamming distance is reported as a second check.

Flagging rule: a cluster is flagged when it has at least ``min_cluster_size`` cases and a
one-sided binomial test says its unsolved share is above the regional share
(p < ``alpha``). Because we test every cluster, we also report a stricter
Bonferroni-corrected flag (p < alpha / k), and, for comparison, the simpler rule we tried
first: a fixed threshold of 50% unsolved.

Validation: the Gary, Indiana case. In 2010 Thomas Hargrove, who later founded the
Murder Accountability Project, flagged a cluster of unsolved strangulations of women in
Lake County, Indiana, with a simple clustering rule. In 2014 a man arrested in Gary
confessed to strangling several women; whether he committed the older unsolved cases is
not proven (Kolker, 2017). We check where those cases end up, for the chosen k and for
every k in the elbow search (sensitivity analysis).

Outputs (in ``reports/metrics``): ``elbow.csv``, ``clusters.csv``,
``cluster_assignments.csv`` and ``gary_check.json``.

Run::

    python -m homicide.cluster                 # region from config.yaml (Indiana)
    python -m homicide.cluster --region Ohio   # any other state
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from scipy.stats import binomtest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder

from homicide.config import load_config, set_seed
from homicide.data import clean, load_raw

log = logging.getLogger(__name__)

GARY_WEAPONS = ["Strangulation", "Suffocation"]


def region_data(cfg: dict, region: str | None = None) -> pd.DataFrame:
    """All years of one region, with the clustering features as strings."""
    c = cfg["clustering"]
    region = region or c["region"]
    df = clean(load_raw(cfg["paths"]["raw_csv"]), cfg["features"]["leakage"])
    df = df[df[c["region_column"]] == region].reset_index(drop=True)
    df[c["features"]] = df[c["features"]].astype(str)
    log.info("%s: %d homicides, unsolved share %.3f", region, len(df), df["y"].mean())
    return df


def fit_kmodes(X: np.ndarray, k: int, n_init: int, seed: int) -> KModes:
    """Fit k-modes with Huang initialisation."""
    km = KModes(n_clusters=k, init="Huang", n_init=n_init, random_state=seed, n_jobs=-1)
    km.fit(X)
    return km


def choose_k(curve: pd.DataFrame) -> int:
    """Elbow = point furthest from the straight line between the first and last point."""
    k = curve["k"].to_numpy(float)
    c = curve["cost"].to_numpy(float)
    kn = (k - k.min()) / (k.max() - k.min())
    cn = (c - c.min()) / (c.max() - c.min())
    dist = np.abs(kn + cn - 1) / np.sqrt(2)   # distance to the line from (0, 1) to (1, 0)
    return int(k[np.argmax(dist)])


def summarise(df: pd.DataFrame, labels: np.ndarray, features: list[str],
              min_size: int, alpha: float) -> pd.DataFrame:
    """One row per cluster: size, unsolved share, p-value, feature modes and flags."""
    base = df["y"].mean()
    k = len(np.unique(labels))
    rows = []
    for cid, g in df.assign(cluster=labels).groupby("cluster"):
        n, u = len(g), int(g["y"].sum())
        row = {"cluster": int(cid), "n_cases": n, "n_unsolved": u, "unsolved_share": u / n,
               "p_value": binomtest(u, n, base, alternative="greater").pvalue,
               "years": f"{g['Year'].min()}-{g['Year'].max()}"}
        for f in features:
            top = g[f].value_counts()
            row[f] = top.index[0]
            row[f"{f} purity"] = float(top.iloc[0] / n)
        rows.append(row)
    table = pd.DataFrame(rows)
    big = table["n_cases"] >= min_size
    table["flagged"] = big & (table["p_value"] < alpha)
    table["flagged_bonferroni"] = big & (table["p_value"] < alpha / k)
    # Simpler rule we tried first: a fixed threshold of 50% unsolved (kept for comparison).
    table["flagged_fixed50"] = big & (table["unsolved_share"] >= 0.5)
    return table.sort_values(["flagged", "p_value"], ascending=[False, True]).reset_index(drop=True)


def gary_mask(df: pd.DataFrame) -> pd.Series:
    """Rows of female strangulation/suffocation victims in Lake County, Indiana."""
    return ((df["City"] == "Lake") & (df["Victim Sex"] == "Female")
            & (df["Weapon"].isin(GARY_WEAPONS)))


def gary_check(df: pd.DataFrame, labels: np.ndarray, table: pd.DataFrame) -> dict:
    """Where do the unsolved female strangulation/suffocation cases of Lake County end up?"""
    target = df[gary_mask(df) & (df["y"] == 1)]
    if target.empty:
        return {"applicable": False}
    lab = pd.Series(labels, index=df.index)[target.index]
    top = int(lab.value_counts().index[0])
    info = table.set_index("cluster").loc[top]
    flagged_ids = set(table.loc[table["flagged"], "cluster"])
    return {
        "applicable": True,
        "unsolved_lake_female_strangulations": int(len(target)),
        "in_any_flagged_cluster": int(lab.isin(flagged_ids).sum()),
        "cluster_with_most_of_them": top,
        "of_them_in_that_cluster": int((lab == top).sum()),
        "cluster_size": int(info["n_cases"]),
        "cluster_unsolved_share": float(info["unsolved_share"]),
        "regional_unsolved_share": float(df["y"].mean()),
        "cluster_p_value": float(info["p_value"]),
        "cluster_flagged": bool(info["flagged"]),
        "cluster_flagged_bonferroni": bool(info["flagged_bonferroni"]),
        "cluster_flagged_fixed50": bool(info["flagged_fixed50"]),
        "in_any_fixed50_cluster": int(lab.isin(set(table.loc[table["flagged_fixed50"],
                                                             "cluster"])).sum()),
        "cluster_profile": {f: info[f] for f in ("City", "Victim Sex", "Victim Age Group",
                                                 "Victim Race", "Weapon") if f in info},
    }


def run(cfg: dict, region: str | None = None, k: int | None = None) -> dict:
    """Elbow search, final clustering, flagging and the Gary check for one region."""
    set_seed(cfg["seed"])
    c = cfg["clustering"]
    region = region or c["region"]
    out_dir = cfg["paths"]["metrics_dir"]
    df = region_data(cfg, region)
    X = df[c["features"]].to_numpy()
    is_indiana = region == "Indiana"

    # Elbow search; for Indiana we also record the Gary check for every k (sensitivity).
    rows, fits = [], {}
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.choice(len(X), size=min(c["silhouette_sample"], len(X)), replace=False)
    X_codes = OrdinalEncoder().fit_transform(X)       # Hamming distance needs codes
    for kk in c["k_range"]:
        km = fit_kmodes(X, kk, c["n_init"], cfg["seed"])
        fits[kk] = km
        row = {"k": kk, "cost": float(km.cost_),
               "silhouette": float(silhouette_score(X_codes[sample], km.labels_[sample],
                                                    metric="hamming"))}
        if is_indiana:
            t = summarise(df, km.labels_, c["features"], c["min_cluster_size"], c["alpha"])
            g = gary_check(df, km.labels_, t)
            row.update({"gary_of_them_in_top_cluster": g["of_them_in_that_cluster"],
                        "gary_in_any_flagged_cluster": g["in_any_flagged_cluster"],
                        "gary_in_any_fixed50_cluster": g["in_any_fixed50_cluster"],
                        "gary_top_cluster_flagged": g["cluster_flagged"]})
        rows.append(row)
        log.info("k=%3d %s", kk, {k_: v for k_, v in row.items() if k_ != "k"})
    curve = pd.DataFrame(rows)
    curve.to_csv(out_dir / "elbow.csv", index=False)

    k = k or choose_k(curve)
    log.info("chosen k = %d", k)
    km = fits.get(k) or fit_kmodes(X, k, c["n_init"], cfg["seed"])
    labels = km.labels_
    table = summarise(df, labels, c["features"], c["min_cluster_size"], c["alpha"])
    table.to_csv(out_dir / "clusters.csv", index=False)
    df.assign(cluster=labels)[["Record ID", "Year", *c["features"], "y", "cluster"]].to_csv(
        out_dir / "cluster_assignments.csv", index=False)

    check = gary_check(df, labels, table) if is_indiana else {"applicable": False}
    check.update({"region": region, "k": k, "n_clusters_flagged": int(table["flagged"].sum()),
                  "n_clusters_flagged_bonferroni": int(table["flagged_bonferroni"].sum()),
                  "n_clusters_flagged_fixed50": int(table["flagged_fixed50"].sum())})
    with open(out_dir / "gary_check.json", "w", encoding="utf-8") as f:
        json.dump(check, f, indent=2, default=str)
    log.info("result: %s", check)
    return check


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--region", default=None, help="value of the region column, e.g. a state")
    parser.add_argument("--k", type=int, default=None, help="skip the elbow choice and use this k")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config), args.region, args.k)


if __name__ == "__main__":
    main()
