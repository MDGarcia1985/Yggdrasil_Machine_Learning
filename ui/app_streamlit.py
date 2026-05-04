"""
app_streamlit.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

Streamlit app entrypoint for the Yggdrasil circuit workflow.

This file is intentionally thin.
UI logic lives in `ui/streamlit_ui/*` modules.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import logging
import streamlit as st

# Configure logging for the application.
# This sets up the root logger once so all module loggers inherit it.
# Guarded to avoid duplicate handlers in framework environments (Streamlit may preconfigure logging).
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

st.set_page_config(page_title="Yggdrasil Circuit Builder", layout="wide")

# Import AFTER Streamlit page config so reruns behave predictably.
from ui.streamlit_ui import render_main_panel

render_main_panel()