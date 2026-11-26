"""Learning curve of the selected model.

Fits the selected model (same hyper-parameters) on growing, stratified samples of the
training years and scores it on that training sample and on the validation years. The gap
between the two curves shows over-fitting; a validation curve that is still rising shows
that more data would help.

Output: ``reports/metrics/learning_curve.csv``. Run (after ``train.py``)::

    python -m homicide.learning_curve
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from homicide.config import load_config, set_seed
from homicide.data import load_splits
from homicide.features import input_columns
from homicide.train import make_estimator, make_pipeline

log = logging.getLogger(__name__)

SIZES = [5_000, 20_000, 60_000, 150_000, None]      # None = the full training set


def run(cfg: dict) -> pd.DataFrame:
    """Fit the selected model on growing samples and score train vs validation."""
    set_seed(cfg["seed"])
    with open(cfg["paths"]["metrics_dir"] / "validation_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    name = summary["selected_model"]
    params = summary[name]["params"]
    train, val, _ = load_splits(cfg)
    cols = input_columns(cfg)
    rows = []
    for size in SIZES:
        if size is None or size >= len(train):
            sub = train
        else:
            sub, _ = train_test_split(train, train_size=size, stratify=train["y"],
                                      random_state=cfg["seed"])
        pipe = make_pipeline(cfg, make_estimator(name, params, cfg["seed"]))
        pipe.fit(sub[cols], sub["y"])
        row = {"train_size": len(sub),
               "train_average_precision": average_precision_score(
                   sub["y"], pipe.predict_proba(sub[cols])[:, 1]),
               "validation_average_precision": average_precision_score(
                   val["y"], pipe.predict_proba(val[cols])[:, 1])}
        rows.append(row)
        log.info("%s", {k: round(v, 4) for k, v in row.items()})
    curve = pd.DataFrame(rows)
    curve.to_csv(cfg["paths"]["metrics_dir"] / "learning_curve.csv", index=False)
    return curve


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
