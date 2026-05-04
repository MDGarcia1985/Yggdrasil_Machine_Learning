# Repository layout

```text
Yggdrasil_Machine_Learning/
|-- LICENSE
|-- README.md
|-- CONTENTS.md
|-- requirements.txt
|-- main.py                 # CLI -> circuit training (same as src.train)
|
|-- Data/
|   |-- raw/
|   |   |-- circuits_sample.csv   # MVP training input
|   |   |-- circuit_sample_two.csv # additional training input
|   |   `-- (legacy ww2 samples, optional)
|   `-- processed/
|
|-- Models/                 # Created by training (gitignored)
|
|-- src/
|   |-- __init__.py
|   |-- preprocess.py
|   |-- features.py
|   |-- feature_importance.py
|   |-- train.py
|   |-- predict.py
|   `-- db/                 # optional DB helpers
|
|-- tests/
|-- ui/                     # Streamlit
|-- config/
|-- docs/
|-- scripts/
|-- utils/
|-- notebooks/
`-- archive/                # old experiments, not part of MVP
```
