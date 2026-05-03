"""
UI package for the Yggdrasil application.

This package exposes high-level Streamlit entrypoints while keeping
module layout details internal.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from .streamlit_ui import render_main_panel, render_sidebar

__all__ = ["render_main_panel", "render_sidebar"]