"""Batch demo: score new homicide cases from a CSV file.

The input CSV uses the same column names as the Kaggle file (extra columns are ignored,
missing optional columns are treated as unknown). The output CSV contains the input
plus three columns:

* ``p_unsolved``         -- predicted probability that the case will not be solved;
* ``p_solved``           -- 1 - p_unsolved;
* ``likely_to_go_cold``  -- True when p_unsolved is above the threshold chosen on validation.

Run::

    python -m homicide.predict examples/sample_input.csv predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from homicide.config import load_config
from homicide.data import clean

log = logging.getLogger(__name__)


NUMERIC_INPUTS = ("Victim Age", "Victim Count", "Year")


class SolvabilityModel:  # pylint: disable=too-few-public-methods
    """Small wrapper around the saved artifact so the CLI and the API share one code path."""

    def __init__(self, path: str | Path):
        art = joblib.load(path)
        self.pipeline = art["pipeline"]
        self.threshold = float(art["threshold"])
        self.columns = art["input_columns"]
        self.name = art["model_name"]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add p_unsolved, p_solved and likely_to_go_cold to a frame of new cases.

        Missing optional columns are filled in as unknown before cleaning, so a CSV with
        only a few columns (for example no victim age) can still be scored.
        """
        X = df.copy()
        for col in self.columns:
            if col not in X.columns:
                X[col] = np.nan if col in NUMERIC_INPUTS else "Unknown"
        X = clean(X)
        p = self.pipeline.predict_proba(X[self.columns])[:, 1]
        out = df.copy()
        out["p_unsolved"] = p.round(4)
        out["p_solved"] = (1 - p).round(4)
        out["likely_to_go_cold"] = p >= self.threshold
        return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    model = SolvabilityModel(cfg["paths"]["models_dir"] / "best_model.joblib")
    df = pd.read_csv(args.input_csv, dtype={"Agency Code": str})
    result = model.predict(df)
    result.to_csv(args.output_csv, index=False)
    log.info("scored %d cases with %s; %d flagged as likely to go cold -> %s",
             len(result), model.name, int(result["likely_to_go_cold"].sum()), args.output_csv)


if __name__ == "__main__":
    main()
