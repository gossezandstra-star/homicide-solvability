"""Reproduce every figure in the report.

Each figure is made by one function below, from files written by the earlier steps
(``data.py``, ``train.py``, ``evaluate.py``, ``cluster.py``). Re-running

    python -m homicide.figures

after the pipeline recreates all PNG files in ``reports/figures`` exactly.

=====  ============================================  =========================================
File   Content                                       Source
=====  ============================================  =========================================
fig01  Unsolved share per year + split               data/processed/*.parquet
fig02  Unsolved share per weapon (training years)    data/processed/train.parquet
fig03  Model comparison on validation                reports/metrics/validation_summary.json
fig04  Precision-recall curves on test               reports/metrics/test_predictions.parquet
fig05  Confusion matrix of selected model on test    reports/metrics/test_predictions.parquet
fig06  Permutation feature importance                reports/metrics/feature_importance.csv
fig07  Bias audit (recall and false-positive rate)   reports/metrics/bias_audit.csv
fig08  Elbow + silhouette of k-modes                 reports/metrics/elbow.csv
fig09  Gary, Indiana check for every k               reports/metrics/elbow.csv
fig10  Drift (PSI) per feature                       reports/metrics/drift_report.csv
fig11  Test performance per year                     reports/metrics/performance_by_year.csv
=====  ============================================  =========================================
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve  # noqa: E402

from homicide.config import load_config  # noqa: E402
from homicide.data import load_splits  # noqa: E402

log = logging.getLogger(__name__)

# Colour roles (fixed order, never cycled). Grey is reserved for the baseline / context.
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
GREY, TEXT, TEXT2, GRID = "#9a9a9a", "#191919", "#666666", "#e6e6e6"
MODEL_LABELS = {
    "majority_baseline": "Majority baseline",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "adaboost": "AdaBoost",
    "hist_gradient_boosting": "Gradient boosting",
}
MODEL_COLORS = {"logistic_regression": SERIES[0], "decision_tree": SERIES[1],
                "random_forest": SERIES[2], "adaboost": SERIES[3],
                "hist_gradient_boosting": SERIES[4], "majority_baseline": GREY}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, cfg, name):
    path = cfg["paths"]["figures_dir"] / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("wrote %s", path.name)


def fig01_unsolved_by_year(cfg):
    """Unsolved share per year, with the train/validation/test split shaded."""
    train, val, test = load_splits(cfg)
    df = pd.concat([train, val, test])
    share = df.groupby("Year")["y"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(share.index, share.values * 100, color=SERIES[0], marker="o", markersize=3)
    s = cfg["split"]
    for lo, hi, label, alpha in ((1979.5, s["train_end_year"] + .5, "train", 0.06),
                                 (s["train_end_year"] + .5, s["val_end_year"] + .5, "val", 0.18),
                                 (s["val_end_year"] + .5, 2014.5, "test", 0.32)):
        ax.axvspan(lo, hi, color=GREY, alpha=alpha, lw=0)
        ax.axvline(hi, color="white", lw=2)
        ax.text((lo + hi) / 2, 34.6, label, ha="center", color=TEXT2, fontsize=9)
    ax.set_ylim(20, 36)
    ax.set_xlim(1979.5, 2014.5)
    ax.set_ylabel("unsolved cases (%)")
    ax.set_title("Share of homicides that were not solved, per year")
    _save(fig, cfg, "fig01_unsolved_by_year")


def fig02_unsolved_by_weapon(cfg):
    """Unsolved share per weapon in the training years."""
    train, _, _ = load_splits(cfg)
    g = train.groupby("Weapon")["y"].agg(["mean", "size"])
    g = g[g["size"] >= 500].sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(g.index, g["mean"] * 100, color=SERIES[0], height=0.7)
    ax.axvline(train["y"].mean() * 100, color=TEXT2, lw=1, ls="--")
    ax.text(train["y"].mean() * 100 + 0.5, -0.9, "overall", color=TEXT2, fontsize=8)
    for i, v in enumerate(g["mean"] * 100):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=8, color=TEXT)
    ax.set_xlabel("unsolved cases (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Unsolved share per weapon (1980-2008, weapons with 500+ cases)")
    _save(fig, cfg, "fig02_unsolved_by_weapon")


def fig03_model_comparison(cfg):
    """Validation metrics of every model."""
    with open(cfg["paths"]["metrics_dir"] / "validation_summary.json", encoding="utf-8") as f:
        s = json.load(f)
    names = ["majority_baseline", *cfg["models"]]
    metrics = [("average_precision", "Average precision"), ("f1", "F1 (unsolved)"),
               ("roc_auc", "ROC AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (m, label) in zip(axes, metrics):
        vals = [s[n][m] for n in names]
        ax.barh([MODEL_LABELS[n] for n in names], vals,
                color=[MODEL_COLORS[n] for n in names], height=0.65)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
        ax.set_xlim(0, 1)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Model comparison on the validation years (2009-2011)",
                 x=0.01, ha="left", fontweight="bold", color=TEXT)
    _save(fig, cfg, "fig03_model_comparison_validation")


def fig04_pr_curves(cfg):
    """Precision-recall curves of every model on the test years."""
    preds = pd.read_parquet(cfg["paths"]["metrics_dir"] / "test_predictions.parquet")
    res = pd.read_csv(cfg["paths"]["metrics_dir"] / "test_results.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for name in cfg["models"]:
        p, r, _ = precision_recall_curve(preds["y"], preds[f"p_{name}"])
        ap = res.loc[name, "average_precision"]
        ax.plot(r, p, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]} (AP {ap:.3f})")
    prev = preds["y"].mean()
    ax.axhline(prev, color=GREY, ls="--", lw=1.5, label=f"Majority baseline (AP {prev:.3f})")
    ax.set_xlabel("recall (share of unsolved cases found)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Precision-recall on the test years (2012-2014)")
    _save(fig, cfg, "fig04_pr_curves_test")


def fig05_confusion(cfg):
    """Confusion matrix of the selected model on the test years."""
    preds = pd.read_parquet(cfg["paths"]["metrics_dir"] / "test_predictions.parquet")
    res = pd.read_csv(cfg["paths"]["metrics_dir"] / "test_results.csv")
    best = res[res["selected"]].iloc[0]
    pred = (preds[f"p_{best['model']}"] >= best["threshold"]).astype(int)
    cm = confusion_matrix(preds["y"], pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["solved", "unsolved"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n({share:.0%} of row)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else TEXT, fontsize=9)
    ax.set_title(f"{MODEL_LABELS[best['model']]}, test years\nthreshold {best['threshold']:.2f}",
                 fontsize=10)
    _save(fig, cfg, "fig05_confusion_matrix_test")


def fig06_importance(cfg):
    """Permutation importance of the selected model."""
    imp = pd.read_csv(cfg["paths"]["metrics_dir"] / "feature_importance.csv")
    imp = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"],
            color=SERIES[0], height=0.65, error_kw={"ecolor": TEXT2, "lw": 1})
    ax.set_xlabel("drop in average precision when the column is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("Permutation importance (validation years)")
    _save(fig, cfg, "fig06_feature_importance")


def fig07_bias(cfg):
    """Recall and false-positive rate per victim group."""
    audit = pd.read_csv(cfg["paths"]["metrics_dir"] / "bias_audit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    for ax, attr in zip(axes, ["Victim Race", "Victim Sex"]):
        a = audit[audit["attribute"] == attr].reset_index(drop=True)
        x = np.arange(len(a))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, a["recall"], w, color=SERIES[0], label="recall (unsolved found)")
        ax.bar(x + w / 2 + 0.01, a["false_positive_rate"], w, color=SERIES[1],
               label="false-positive rate")
        ax.set_xticks(x, [g.replace("/", "/\n") for g in a["group"]], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(attr, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Bias audit of the selected model on the test years",
                 x=0.01, ha="left", fontweight="bold", color=TEXT)
    _save(fig, cfg, "fig07_bias_audit")


def fig08_elbow(cfg):
    """k-modes cost per k, with the chosen k."""
    curve = pd.read_csv(cfg["paths"]["metrics_dir"] / "elbow.csv")
    with open(cfg["paths"]["metrics_dir"] / "gary_check.json", encoding="utf-8") as f:
        k = json.load(f)["k"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, label in ((axes[0], "cost", "k-modes cost (elbow method)"),
                           (axes[1], "silhouette", "silhouette score (Hamming distance)")):
        ax.plot(curve["k"], curve[col], color=SERIES[0], marker="o", markersize=5)
        y = float(curve.loc[curve["k"] == k, col].iloc[0])
        ax.scatter([k], [y], s=120, facecolor="none", edgecolor=TEXT, lw=1.5, zorder=3)
        ax.annotate(f"chosen k = {k}", (k, y), xytext=(10, 12), textcoords="offset points",
                    color=TEXT, fontsize=9)
        ax.set_xlabel("number of clusters k")
        ax.set_title(label, fontsize=10)
    fig.suptitle(f"Choosing k for k-modes, {cfg['clustering']['region']}",
                 x=0.01, ha="left", fontweight="bold", color=TEXT)
    _save(fig, cfg, "fig08_kmodes_elbow")


def fig09_gary(cfg):
    """How many Gary cases land in a flagged cluster, per k."""
    curve = pd.read_csv(cfg["paths"]["metrics_dir"] / "elbow.csv")
    if "gary_in_any_flagged_cluster" not in curve:
        return
    with open(cfg["paths"]["metrics_dir"] / "gary_check.json", encoding="utf-8") as f:
        g = json.load(f)
    total = g["unsolved_lake_female_strangulations"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [SERIES[0] if kk == g["k"] else "#9cc3ef" for kk in curve["k"]]
    ax.bar(curve["k"].astype(str), curve["gary_in_any_flagged_cluster"], color=colors, width=0.6)
    for i, v in enumerate(curve["gary_in_any_flagged_cluster"]):
        ax.text(i, v + 0.3, f"{v}/{total}", ha="center", fontsize=8, color=TEXT)
    ax.set_ylim(0, total + 2)
    ax.set_xlabel("number of clusters k (chosen k highlighted)")
    ax.set_ylabel("cases in a flagged cluster")
    ax.grid(axis="x", visible=False)
    ax.set_title("Gary, Indiana: unsolved strangulations of women\n"
                 "in Lake County that land in a flagged cluster", fontsize=10)
    _save(fig, cfg, "fig09_gary_check_per_k")


def fig10_drift(cfg):
    """Population Stability Index per feature, train vs test years."""
    rep = pd.read_csv(cfg["paths"]["metrics_dir"] / "drift_report.csv").sort_values("psi")
    m = cfg["monitoring"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(rep["feature"], rep["psi"], color=SERIES[0], height=0.65)
    for x, label in ((m["psi_warning"], "moderate"), (m["psi_alert"], "significant")):
        ax.axvline(x, color=TEXT2, lw=1, ls="--")
        ax.text(x, len(rep) - 0.4, f" {label}", color=TEXT2, fontsize=8, va="bottom")
    for i, v in enumerate(rep["psi"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax.set_xlabel("Population Stability Index (training years vs test years)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Data and prediction drift")
    _save(fig, cfg, "fig10_drift_psi")


def fig11_performance_by_year(cfg):
    """Test performance per year."""
    perf = pd.read_csv(cfg["paths"]["metrics_dir"] / "performance_by_year.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for col, color, label in (("average_precision", SERIES[0], "average precision"),
                              ("f1", SERIES[1], "F1 (unsolved)"),
                              ("recall", SERIES[2], "recall (unsolved)")):
        ax.plot(perf["year"], perf[col], color=color, marker="o", markersize=6, label=label)
    ax.set_xticks(perf["year"])
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Performance of the selected model per test year")
    _save(fig, cfg, "fig11_performance_by_year")


ALL = [fig01_unsolved_by_year, fig02_unsolved_by_weapon, fig03_model_comparison,
       fig04_pr_curves, fig05_confusion, fig06_importance, fig07_bias, fig08_elbow, fig09_gary,
       fig10_drift, fig11_performance_by_year]


def run(cfg):
    """Recreate every figure."""
    for f in ALL:
        f(cfg)


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
