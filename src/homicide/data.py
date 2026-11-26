"""Loading, cleaning and splitting the Homicide Reports dataset.

Dataset: Murder Accountability Project, *Homicide Reports, 1980-2014* (Kaggle).
One row is one victim of a homicide reported to the FBI (or obtained through FOIA).

Run as a script to write the cleaned splits to ``data/processed``::

    python -m homicide.data
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from homicide.config import load_config

log = logging.getLogger(__name__)

TARGET = "Crime Solved"
# Positive class = the case is NOT solved (at risk of going cold).
POSITIVE_LABEL = "No"

AGE_UNKNOWN = 998          # code used by the SHR for an unknown victim age
AGE_BINS = [-1, 0, 12, 17, 24, 34, 49, 64, 200]
AGE_LABELS = ["infant", "1-12", "13-17", "18-24", "25-34", "35-49", "50-64", "65+"]


def load_raw(path) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded from Kaggle."""
    df = pd.read_csv(path, low_memory=False, dtype={"Agency Code": str, "Perpetrator Age": str})
    log.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    return df


def clean(df: pd.DataFrame, leakage_cols: list[str] | None = None) -> pd.DataFrame:
    """Clean the raw data.

    * drops columns that are only known after a case is solved (label leakage);
    * turns the "unknown age" code 998 into a missing value;
    * adds a binary label ``y`` (1 = unsolved) and an age-group column used for clustering.
    """
    df = df.copy()
    if leakage_cols:
        df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

    df["Victim Age"] = df["Victim Age"].replace(AGE_UNKNOWN, np.nan).astype(float)
    df["Victim Age Group"] = pd.cut(df["Victim Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    df.loc[df["Victim Age"].isna(), "Victim Age Group"] = "Unknown"

    if TARGET in df.columns:          # new, unlabelled cases have no target column
        df["y"] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return df


def time_split(df: pd.DataFrame, train_end: int, val_end: int):
    """Split by year: train <= train_end < validation <= val_end < test."""
    train = df[df["Year"] <= train_end]
    val = df[(df["Year"] > train_end) & (df["Year"] <= val_end)]
    test = df[df["Year"] > val_end]
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("%-5s %7d rows, %d-%d, unsolved share %.3f",
                 name, len(part), part["Year"].min(), part["Year"].max(), part["y"].mean())
    return train, val, test


def load_splits(cfg: dict):
    """Load the processed splits written by :func:`main` (or build them on the fly)."""
    out = cfg["paths"]["processed_dir"]
    files = [out / f"{s}.parquet" for s in ("train", "val", "test")]
    if all(f.exists() for f in files):
        return tuple(pd.read_parquet(f) for f in files)
    return build_splits(cfg, save=True)


def build_splits(cfg: dict, save: bool = True):
    """Run the full load -> clean -> split step."""
    df = clean(load_raw(cfg["paths"]["raw_csv"]), cfg["features"]["leakage"])
    splits = time_split(df, cfg["split"]["train_end_year"], cfg["split"]["val_end_year"])
    if save:
        for name, part in zip(("train", "val", "test"), splits):
            part.to_parquet(cfg["paths"]["processed_dir"] / f"{name}.parquet", index=False)
    return splits


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_splits(load_config(args.config), save=True)


if __name__ == "__main__":
    main()
