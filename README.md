# Yggdrasil Machine Learning

MVP pipeline for **circuit row data**: clean tabular exports, engineer features, train a classifier for the next component type, and persist artifacts for inference.

## Requirements

Python 3.11+ recommended. Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you are not using the project venv, use the same Python executable that will
run the app.

## Run training (default sample)

Uses the configured training CSV files:

- `Data/raw/circuits_sample.csv`
- `Data/raw/circuit_sample_two.csv`

Training writes under `Models/` (`model.pkl`, `encoder.pkl`, `feature_columns.pkl`, `training_report.pkl`).

```powershell
.\venv\Scripts\python.exe main.py
```

or:

```powershell
.\venv\Scripts\python.exe -m src.train
```

## Run inference

After training:

```powershell
.\venv\Scripts\python.exe -m src.predict
```

## Streamlit UI

Direct Streamlit launch:

```powershell
.\venv\Scripts\python.exe -m streamlit run ui/app_streamlit.py
```

Local launcher, which opens the browser after starting Streamlit:

```powershell
.\venv\Scripts\python.exe ui/launcher.py
```

Pipeline-first launcher:

```powershell
.\venv\Scripts\python.exe scripts/app/run_app.py
```

Skip retraining and launch only the app:

```powershell
.\venv\Scripts\python.exe scripts/app/run_app.py --skip-pipeline
```

## Layout

| Path | Role |
|------|------|
| `Data/raw/` | Input CSVs (`circuits_sample.csv`, `circuit_sample_two.csv`) |
| `src/preprocess.py` | Load, clean, engineer, labels |
| `src/features.py` | Feature matrix and targets |
| `src/train.py` | Train, evaluate, save artifacts |
| `src/predict.py` | Load artifacts and predict |
| `src/graph.py`, `src/nodes.py`, `src/edges.py`, `src/simulator.py` | Circuit graph model and lightweight validation |
| `ui/` | Streamlit app |
| `tests/` | Pytest smoke tests |

Legacy WWII aircraft sample files under `Data/` are not used by the circuit MVP. See `CONTENTS.md` for a fuller tree.

## License

MIT — see `LICENSE`.
