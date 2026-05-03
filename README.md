# WWII Aircraft ML

WWII Aircraft ML is a small machine learning project that cleans a historical aircraft dataset, engineers numeric features, trains a classifier, and predicts whether an aircraft is more likely to belong to the `Fighter` or `Bomber` class.

The repository includes:

- an Excel Office Script for fixing semicolon-delimited spreadsheet imports
- a Python preprocessing pipeline built with `pandas` and `NumPy`
- feature preparation and model training with `scikit-learn`
- model persistence with `joblib`
- lightweight test scripts for each project stage

## Project Goal

The goal of the project is to take raw WWII aircraft records and move them through a reproducible workflow:

1. Repair spreadsheet rows when Excel imports the dataset incorrectly.
2. Clean and normalize numeric aircraft fields.
3. Engineer model-ready features.
4. Simplify historical aircraft roles into broader labels.
5. Train a classification model.
6. Save the trained model for later prediction.

## Main Dataset Reference

Primary dataset reference:

Iskk97. (n.d.). *WW2 aircraft* [Data set]. Kaggle. Retrieved March 23, 2026, from https://www.kaggle.com/datasets/iskk97/ww2-aircraft

## Repository Layout

```text
ww2-aircraft-ml/
|-- LICENSE
|-- CONTENTS.md
|-- README.md
|-- main.py
|
|-- Data/
|   |-- raw/
|   |   |-- ww2aircraft.csv
|   |   `-- WW2 Planes Dataset.xlsx
|   `-- processed/
|       |-- aircraft_cleaned.csv
|       `-- ww2aircraft.xlsm
|
|-- src/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- features.py
|   |-- train.py
|   `-- predict.py
|
|-- tests/
|   |-- __init__.py
|   |-- test_preprocess.py
|   |-- test_features.py
|   |-- test_train.py
|   |-- test_predict.py
|   `-- data/
|       `-- test_data.txt
|
|-- scripts/
|   `-- excel/
|       |-- README.md
|       `-- parse_semicolon_data.ts
|
|-- utils/
|   |-- __init__.py
|   `-- run_tests.py
|
|-- models/
|   `-- model.pkl
|
|-- notebooks/
|   `-- exploration.ipynb
|
`-- archive/
    |-- functions.txt
    |-- main.py
    `-- utils.py
```

## Tech Stack

- Python 3
- pandas
- NumPy
- scikit-learn
- joblib
- Microsoft Excel Office Scripts

## How the Pipeline Works

### 1. Excel repair step

If the original CSV imports badly into Excel, use the Office Script in [scripts/excel/parse_semicolon_data.ts](Machine_Learning\scripts\excel\parse_semicolon_data.ts) to reconstruct rows and split semicolon-delimited values into the correct columns.

See the Excel-specific instructions in [scripts/excel/README.md](Machine_Learning\scripts\excel\README.md).

Important:

- If you use Excel Desktop, save the workbook as `.xlsm` before working with the script workflow.
- Export the repaired sheet to CSV before running the Python pipeline.

### 2. Preprocessing

The preprocessing stage in [src/preprocess.py](Machine_Learning\src\preprocess.py) does the following:

- loads the cleaned CSV
- selects required aircraft columns
- converts numeric text values into usable numeric data
- removes rows missing critical model fields
- creates derived features such as `AspectRatio`, `SizeIndex`, and `NumberBuilt_log`
- maps historical role labels into broader classes

### 3. Feature preparation

[src/features.py](Machine_Learning\src\features.py) builds:

- `X`: the feature matrix used for model input
- `y`: the target labels from `RoleClass`

The project currently uses these model features:

- `Crew`
- `Length`
- `Wingspan`
- `Height`
- `WingArea`
- `MaxSpeed`
- `AspectRatio`
- `SizeIndex`
- `NumberBuilt_log`

### 4. Training and evaluation

[src/train.py](Machine_Learning\src\train.py) handles:

- train/test splitting with `train_test_split`
- feature scaling with `StandardScaler`
- classification with `LogisticRegression`
- evaluation using accuracy, classification report, and confusion matrix
- saving the model artifact to `models/model.pkl`

### 5. Prediction

[src/predict.py](Machine_Learning\src\predict.py) loads the saved model and predicts the class for one aircraft record using the same feature order used during training.

## Prerequisites

Before running the project, make sure you have:

- Python installed
- a local virtual environment
- the project dependencies installed into that virtual environment
- the cleaned aircraft CSV available in the processed data folder

## Setup

### Windows PowerShell setup

Create and activate a virtual environment:

```powershell
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Install the libraries used by the project:

```powershell
python -m pip install pandas numpy scikit-learn joblib
```

If you want to verify the installed packages:

```powershell
python -m pip show pandas numpy scikit-learn joblib
```

## Data Preparation Workflow

Use this sequence when starting from the raw dataset:

1. Download the dataset from Kaggle.
2. Open the raw file in Excel if the CSV structure is broken.
3. Run the Office Script in `scripts/excel/parse_semicolon_data.ts`.
4. Save the workbook as `.xlsm` if you are using Excel Desktop.
5. Export the cleaned data to CSV.
6. Place the cleaned CSV in `Data/processed/`.
7. Run the Python pipeline.

## Running the Project

From the project root:

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

The main pipeline script:

- loads processed aircraft data
- preprocesses and engineers features
- builds the feature matrix and target labels
- trains the classifier
- evaluates the model
- saves the model artifact to `models/model.pkl`

## Command Reference

### Run the full pipeline

```powershell
python main.py
```

### Run the preprocessing sniff test

```powershell
python -m tests.test_preprocess
```

### Run the feature-preparation sniff test

```powershell
python -m tests.test_features
```

### Run the training sniff test

```powershell
python -m tests.test_train
```

### Run the prediction sniff test

```powershell
python -m tests.test_predict
```

### View the current test log

```powershell
Get-Content tests\data\test_data.txt
```

## Prediction Example

After a model has been trained and saved, you can call the prediction module from Python:

```python
import numpy as np
from src.predict import predict_aircraft

sample = {
    "Crew": 1,
    "Length": 9.83,
    "Wingspan": 11.28,
    "Height": 3.71,
    "WingArea": 21.8,
    "MaxSpeed": 710,
    "AspectRatio": (11.28 ** 2) / 21.8,
    "SizeIndex": 9.83 * 11.28,
    "NumberBuilt_log": np.log1p(15875),
}

result = predict_aircraft(sample)
print(result)
```

The prediction result includes:

- `prediction`
- `probabilities`
- `confidence`
- `feature_columns`

## Notes on Paths and Portability

The repository currently stores data under `Data/`, while some Python modules reference `data/processed/...`.

On Windows, this usually works because file paths are case-insensitive. On case-sensitive systems, you may need to rename the folder or update the file paths in the Python modules before running the pipeline.

## Test Logging

The test scripts in `tests/` are lightweight execution checks rather than a formal `pytest` suite. Each script captures console output and appends the results to:

```text
tests/data/test_data.txt
```

This makes it easier to keep a simple chronological record of manual test runs during development.

## Key Files

- [main.py](Machine_Learning\main.py): runs the end-to-end pipeline
- [src/preprocess.py](Machine_Learning\src\preprocess.py): data loading, cleaning, and feature engineering
- [src/features.py](Machine_Learning\src\features.py): model feature selection and matrix construction
- [src/train.py](Machine_Learning\src\train.py): splitting, scaling, training, evaluation, and persistence
- [src/predict.py](Machine_Learning\src\predict.py): single-record prediction from a saved model
- [utils/run_tests.py](Machine_Learning\utils\run_tests.py): output capture and test log writer
- [scripts/excel/parse_semicolon_data.ts](Machine_Learning\scripts\excel\parse_semicolon_data.ts): Excel Office Script for row repair

## References

Primary data source:

- Iskk97. (n.d.). *WW2 aircraft* [Data set]. Kaggle. Retrieved March 23, 2026, from https://www.kaggle.com/datasets/iskk97/ww2-aircraft

Tools and documentation:

- Python Software Foundation. (n.d.). *Python documentation*. https://www.python.org/doc/
- NumPy Developers. (n.d.). *NumPy documentation*. https://numpy.org/doc/
- The pandas development team. (n.d.). *pandas documentation*. https://pandas.pydata.org/docs/
- scikit-learn developers. (n.d.). *scikit-learn documentation*. https://scikit-learn.org/stable/
- Joblib developers. (n.d.). *Joblib documentation*. https://joblib.readthedocs.io/en/stable/
- Microsoft. (n.d.). *Office Scripts in Excel*. https://learn.microsoft.com/office/dev/scripts/
- Kaggle. (n.d.). *Kaggle datasets*. https://www.kaggle.com/datasets

## License

This project is released under the MIT License. See [LICENSE](Machine_Learning\LICENSE).
