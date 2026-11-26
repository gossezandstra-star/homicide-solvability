"""Predicting unsolved homicides and detecting serial-offender patterns.

Modules
-------
config    -- loads ``config.yaml`` and resolves paths relative to the project root.
data      -- loads the raw CSV, cleans it and makes the time-based split.
features  -- builds the scikit-learn preprocessing pipeline.
train     -- trains the baseline and the classical models, tunes them on validation.
evaluate  -- evaluates the selected model once on the test set.
cluster   -- k-modes cluster detector for possible serial-offender patterns.
figures   -- reproduces every figure used in the report.
predict   -- batch demo: CSV in, predictions out.
"""

__version__ = "1.0.0"
