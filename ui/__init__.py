"""
UI package for the Yggdrasil application.

This package exposes high-level Streamlit entrypoints while keeping
module layout details internal.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

__all__ = ["render_main_panel", "render_sidebar"]


def __getattr__(name):
    """
    Lazily expose Streamlit entrypoints without requiring Streamlit for every
    import of the ui package.
    """
    if name == "render_main_panel":
        from .streamlit_ui import render_main_panel

        return render_main_panel

    if name == "render_sidebar":
        from .streamlit_ui import render_sidebar

        return render_sidebar

    raise AttributeError(f"module 'ui' has no attribute {name!r}")
