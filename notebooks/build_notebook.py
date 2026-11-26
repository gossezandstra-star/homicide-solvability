"""Generates ``homicide_pipeline.ipynb``: a readable walkthrough that imports the code
from ``src/homicide`` instead of duplicating it. Run from the project root::

    python notebooks/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Predicting unsolved homicides and detecting serial-offender patterns\n"
       "61.501 Production Ready Machine Learning - project notebook.\n\n"
       "This notebook walks through the full pipeline. All logic lives in `src/homicide/`; "
       "the notebook only calls it and shows the results. Every step can also be run from "
       "the command line (`make all`, see README).\n\n"
       "**Before running:** put `database.csv` in `data/raw/` (see `data/README.md`)."),
    code("import sys, json\nfrom pathlib import Path\n"
         "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
         "sys.path.insert(0, str(ROOT / 'src'))\n"
         "import pandas as pd\nfrom IPython.display import Image, display\n"
         "from homicide.config import load_config\n"
         "from homicide import data, train, evaluate, learning_curve, monitor, cluster, figures\n"
         "cfg = load_config(ROOT / 'config.yaml')\nFIG = cfg['paths']['figures_dir']\n"
         "MET = cfg['paths']['metrics_dir']"),
    md("## 1. Data: load, clean, split by year\n"
       "Perpetrator fields and `Relationship` are only known for solved cases, so they are "
       "dropped (label leakage). We train on 1980-2008, tune on 2009-2011 and test once on "
       "2012-2014."),
    code("train_df, val_df, test_df = data.build_splits(cfg, save=True)\n"
         "pd.DataFrame({s: {'rows': len(d), 'years': f\"{d.Year.min()}-{d.Year.max()}\", "
         "'unsolved share': round(d.y.mean(), 3)} for s, d in "
         "zip(['train', 'validation', 'test'], [train_df, val_df, test_df])}).T"),
    md("## 2. Train and tune the classifiers\n"
       "Majority baseline vs. logistic regression, decision tree, random forest, AdaBoost "
       "and gradient boosting. Positive class = *unsolved*. Hyper-parameters are chosen on "
       "the validation years by average precision. The heavy steps below are switched off "
       "by default and reuse the saved outputs; set `RERUN = True` to recompute everything "
       "(about 50 minutes on 2 CPU cores)."),
    code("RERUN = False\n"
         "if RERUN or not (cfg['paths']['models_dir'] / 'best_model.joblib').exists():\n"
         "    train.run(cfg)\n"
         "val_summary = json.load(open(MET / 'validation_summary.json'))\n"
         "print('selected model:', val_summary['selected_model'])\n"
         "pd.read_csv(MET / 'tuning_results.csv').sort_values('average_precision', "
         "ascending=False).groupby('model').head(1).round(3)"),
    md("## 3. Evaluate once on the test years\n"
       "Test metrics per model, paired bootstrap intervals, calibration and the learning "
       "curve."),
    code("if RERUN:\n    evaluate.run(cfg)\n    learning_curve.run(cfg)\n"
         "pd.read_csv(MET / 'test_results.csv').round(3)"),
    code("pd.read_csv(MET / 'bootstrap_ci.csv').round(4)"),
    code("print(json.load(open(MET / 'calibration_summary.json')))\n"
         "pd.read_csv(MET / 'learning_curve.csv').round(3)"),
    md("## 4. Bias audit, fairer thresholds and error examples"),
    code("pd.read_csv(MET / 'bias_audit.csv').round(3)"),
    code("pd.read_csv(MET / 'fairness_thresholds.csv').round(3)"),
    code("pd.read_csv(MET / 'error_examples.csv')"),
    md("## 5. Drift monitoring\nTraining years vs test years (KS, chi-square, PSI)."),
    code("if RERUN:\n    monitor.run(cfg)\n"
         "pd.read_csv(MET / 'drift_report.csv').round(4)"),
    md("## 6. Cluster detector (k-modes) and the Gary, Indiana check"),
    code("if RERUN:\n    cluster.run(cfg)\n"
         "pd.read_csv(MET / 'elbow.csv').round(3)"),
    code("json.load(open(MET / 'gary_check.json'))"),
    code("pd.read_csv(MET / 'clusters.csv').query('flagged')"),
    md("## 7. Figures\nAll figures are created by `figures.py` from the saved metrics."),
    code("figures.run(cfg)\nfor f in sorted(FIG.glob('fig*.png')):\n    display(Image(filename=str(f)))"),
    md("## 8. Demo: score new cases"),
    code("from homicide.predict import SolvabilityModel\n"
         "model = SolvabilityModel(cfg['paths']['models_dir'] / 'best_model.joblib')\n"
         "model.predict(pd.read_csv(ROOT / 'examples' / 'sample_input.csv', "
         "dtype={'Agency Code': str}))"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).with_name("homicide_pipeline.ipynb")
nbf.write(nb, out)
print("wrote", out)
