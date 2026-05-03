# DEV NOTES

Last updated: 2026-05-03

## Current Execution Model

The project now runs through `main.py`.

Canonical flow:
1. Load and preprocess `Data/raw/circuits_sample.csv`
2. Train model (`next_component_type` target)
3. Save artifacts to `Models/`
4. Run a small prediction preview

Primary command:

```powershell
.\venv\Scripts\python.exe main.py
```

## Key Entrypoints

- `main.py`
  - Canonical MVP entrypoint for training + artifact save + prediction preview.

- `src/train.py`
  - Core training implementation used by `main.py`.

- `src/predict.py`
  - Inference helpers for single-row and DataFrame prediction.

- `scripts/app/run_app.py`
  - App launcher script.
  - Runs `main.py` first by default, then launches Streamlit.
  - Use `--skip-pipeline` to launch app without retraining.

- `ui/app_streamlit.py`
  - Current Streamlit app entrypoint.
  - Imports `render_main_panel()` from `ui.streamlit_ui`.

- `ui/launcher.py`
  - Local browser-opening Streamlit launcher.
  - Launches `ui/app_streamlit.py` through the current Python executable.

## Path Conventions

Shared paths are defined in `utils/paths.py`:
- project root
- data directories
- model artifact paths
- docs/test log paths
- app and main entrypoint paths

Use this module in scripts instead of hardcoding relative paths.

Path constants now include:
- `PROJECT_ROOT`
- `RAW_CIRCUITS_SAMPLE`
- `MAIN_ENTRYPOINT`
- `STREAMLIT_APP_PATH`
- `TEST_LOG_PATH`
- docs/model artifact paths

## Current Data/Artifact Locations

- Input sample: `Data/raw/circuits_sample.csv`
- Artifacts:
  - `Models/model.pkl`
  - `Models/encoder.pkl`
  - `Models/feature_columns.pkl`
  - `Models/training_report.pkl`
- Test log:
  - `tests/data/test_data.txt`

## Testing

Run all tests:

```powershell
.\venv\Scripts\python.exe -m pytest tests
```

Run path-only checks:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_paths.py
```

## Streamlit Run Commands

Install dependencies into the same environment used to launch the app:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Direct Streamlit launch:

```powershell
.\venv\Scripts\python.exe -m streamlit run ui/app_streamlit.py
```

Browser-opening launcher:

```powershell
.\venv\Scripts\python.exe ui/launcher.py
```

Pipeline-first launcher:

```powershell
.\venv\Scripts\python.exe scripts/app/run_app.py
```

Skip retraining:

```powershell
.\venv\Scripts\python.exe scripts/app/run_app.py --skip-pipeline
```

Test output behavior:
- Prints readable, bulleted details to terminal
- Appends the same run details to `tests/data/test_data.txt`

## Streamlit Integration Notes

Current Streamlit wiring:
- `ui/app_streamlit.py` -> `ui.streamlit_ui.render_main_panel()`
- `ui/streamlit_ui/main_panel.py` imports:
  - `render_sidebar` from `ui.streamlit_ui.sidebar_ui`
  - `render_circuit_builder` from `ui.streamlit_ui.circuit_builder`
- `ui/streamlit_ui/circuit_builder.py` builds `src.graph.CircuitGraph`
  from session components and runs `src.simulator.tick_sim()`.
- graph rendering uses:
  - `ui.streamlit_ui.node_styling`
  - `ui.streamlit_ui.edge_styling`
  - `ui.streamlit_ui.graph_styling`

Package exports:
- `ui/streamlit_ui/__init__.py` exports:
  - `render_main_panel`
  - `render_sidebar`
- `ui/__init__.py` exports:
  - `render_main_panel`
  - `render_sidebar`

Sidebar behavior:
- Training action calls `src.train.train_model()` and `save_artifacts()`.
- Prediction action requires model artifacts to exist (`Models/model.pkl`).

## Config Integration Notes

`config/TrainingUIConfig` remains the stable exported type.

`config/ui_config.py` conversion helpers currently return serializable dict payloads:
- `to_grid_config()`
- `to_q_config()`

Reason:
- Avoid unresolved legacy import paths from prior project layouts.
- Keep UI configuration transport-friendly while preserving method names.

## Documentation Map

- Workflow walkthrough: `docs/WORKFLOW_NOTES.md`
- Commenting rules: `docs/COMMENTING_POLICY.md`
- Source overview: `src/README.md`
