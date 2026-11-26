# Data

The dataset is **not** included in this repository (it is 112 MB).

**Homicide Reports, 1980-2014** - Murder Accountability Project, licence CC BY-SA 4.0
<https://www.kaggle.com/datasets/murderaccountability/homicide-reports>

## Download

Option 1 - browser: log in to Kaggle, click *Download*, unzip and place the file at

```
data/raw/database.csv
```

Option 2 - Kaggle API (needs a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
pip install kaggle
kaggle datasets download -d murderaccountability/homicide-reports -p data/raw --unzip
```

Check: `data/raw/database.csv` has 638,454 rows and 24 columns.

## What is in it

One row per homicide victim reported to the FBI's Supplementary Homicide Report (SHR),
plus about 22,000 cases obtained through Freedom of Information Act requests.
The label is `Crime Solved` (Yes/No); about 30% of cases are unsolved.

Columns about the perpetrator (`Perpetrator *`, `Relationship`) are only filled in when a
case is solved, so they are **dropped** before modelling (label leakage).
