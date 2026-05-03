"""
main_panel.py

Purpose:
Main Streamlit landing panel for the Yggdrasil ML circuit application.

Design:
Keep this file focused on page structure and high-level workflow.
Circuit entry logic lives in circuit_builder.py.
Sidebar controls live in sidebar_ui.py.
"""

from __future__ import annotations

import streamlit as st

from ui.streamlit_ui.circuit_builder import render_circuit_builder
from ui.streamlit_ui.sidebar_ui import render_sidebar


def render_main_panel() -> None:
    """
    Render the primary Streamlit application panel.
    """
    sidebar_state = render_sidebar()

    st.title("Yggdrasil Circuit Builder")
    st.caption("Circuit database input, ML prediction, validation, and reinforcement workflow.")

    st.markdown(
        """
        This MVP lets a user create a circuit, add components, assign component pins
        to nets, and prepare circuit data for the machine-learning pipeline.
        """
    )

    selected_mode = sidebar_state.get("selected_mode", "New Circuit")
    graph_view = sidebar_state.get("graph_view", "Concept")

    if selected_mode == "New Circuit":
        render_circuit_builder(graph_view=graph_view)

    elif selected_mode == "New Primitive":
        st.subheader("New Primitive")
        st.info("Primitive creation is planned for a later editor panel.")

    elif selected_mode == "New Structure":
        st.subheader("New Structure")
        st.info("Structure creation is planned for a later editor panel.")

    elif selected_mode == "New Functional Block":
        st.subheader("New Functional Block")
        st.info("Functional block creation is planned for a later editor panel.")

    elif selected_mode == "New Part":
        st.subheader("New Part")
        st.info("Part creation is planned for a later editor panel.")

    else:
        st.warning("Unknown mode selected.")


if __name__ == "__main__":
    render_main_panel()
