# Yggdrasil Machine Learning

MVP pipeline for **circuit row data**: clean tabular exports, engineer features, train a classifier for the next component type, and persist artifacts for inference.

## Requirements

Python 3.11+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run training (default sample)

Uses `Data/raw/circuits_sample.csv` and writes under `Models/` (`model.pkl`, `encoder.pkl`, `feature_columns.pkl`, `training_report.pkl`).

```bash
python main.py
```

or:

```bash
python -m src.train
```

## Run inference

After training:

```bash
python -m src.predict
```

## Streamlit UI

```bash
streamlit run ui/app_streamlit.py
```

## Layout

| Path | Role |
|------|------|
| `Data/raw/` | Input CSV (sample: `circuits_sample.csv`) |
| `src/preprocess.py` | Load, clean, engineer, labels |
| `src/features.py` | Feature matrix and targets |
| `src/train.py` | Train, evaluate, save artifacts |
| `src/predict.py` | Load artifacts and predict |
| `ui/` | Streamlit app |
| `tests/` | Pytest smoke tests |

Legacy WWII aircraft sample files under `Data/` are not used by the circuit MVP. See `CONTENTS.md` for a fuller tree.

## License

MIT — see `LICENSE`.
