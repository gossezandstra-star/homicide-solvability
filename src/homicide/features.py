"""Feature engineering: turns cleaned rows into a numeric matrix.

The preprocessing is a scikit-learn ``ColumnTransformer`` so that it is fitted on the
training data only and then applied unchanged to validation, test and new cases.

* categorical columns  -> one-hot encoding (rare categories grouped together);
* numeric columns      -> median imputation + missing-value flag + standardisation;
* agency and city      -> target encoding: the (smoothed) share of unsolved cases of that
  agency/city in the training years. ``TargetEncoder`` uses cross-fitting during
  ``fit_transform``, so a row never sees its own label.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


def input_columns(cfg: dict) -> list[str]:
    """All raw columns the model needs as input."""
    f = cfg["features"]
    return f["categorical"] + f["numeric"] + f["target_encoded"]


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Create an unfitted preprocessing step from the feature lists in the config."""
    f = cfg["features"]
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                sparse_output=False, dtype=np.float32)
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    target = TargetEncoder(target_type="binary",
                           cv=KFold(n_splits=5, shuffle=True, random_state=cfg["seed"]))
    return ColumnTransformer(
        [
            ("cat", categorical, f["categorical"]),
            ("num", numeric, f["numeric"]),
            ("te", target, f["target_encoded"]),
        ],
        verbose_feature_names_out=False,
    )
