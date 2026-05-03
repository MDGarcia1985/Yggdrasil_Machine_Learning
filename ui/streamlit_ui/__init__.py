"""
streamlit_ui package

Streamlit UI modules for the Yggdrasil circuit application.

Design intent:
- Keep app_streamlit.py thin
- Keep UI helpers internal unless explicitly needed
- Expose only high-level entry points
"""

__all__ = [
    "render_sidebar",
    "render_main_panel"
]


def __getattr__(name):
    """
    Lazily expose Streamlit render functions so non-Streamlit helpers in this
    package remain importable in tests and scripts.
    """
    if name == "render_sidebar":
        from .sidebar_ui import render_sidebar

        return render_sidebar

    if name == "render_main_panel":
        from .main_panel import render_main_panel

        return render_main_panel

    raise AttributeError(f"module 'ui.streamlit_ui' has no attribute {name!r}")
