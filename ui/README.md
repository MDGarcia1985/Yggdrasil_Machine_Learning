# UI Run Instructions

Streamlit entrypoint:

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

The UI renders through `ui/app_streamlit.py`, then delegates to
`ui.streamlit_ui.render_main_panel()`.
