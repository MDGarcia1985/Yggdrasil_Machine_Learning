"""
sidebar_ui.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

Purpose:
Sidebar controls for the Yggdrasil Streamlit UI.

Design:
The sidebar provides:
- navigation for Add modes
- help text for selected mode
- ML action buttons for train, predict, and save/export workflows

Notes:
The train/predict/save functions are intentionally lightweight wrappers.
They call into the main application modules when available and fail safely
with Streamlit messages during MVP development.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


MODEL_PATH = Path("Models/model.pkl")


ADD_OPTIONS = {
    "New Circuit": "Create a full circuit using primitives, structures, functional blocks, parts, and custom entries.",
    "New Primitive": "Add a fundamental electrical element such as a resistor, capacitor, diode, transistor, or fuse.",
    "New Structure": "Create a reusable circuit pattern such as a voltage divider, RC filter, pull-up, or transistor switch.",
    "New Functional Block": "Create a reusable functional circuit unit such as a 555 timer, op-amp, comparator, motor driver, MCU, or FPGA.",
    "New Part": "Add a real-world purchasable component with manufacturer, package, footprint, and part-number data.",
}


EXPORT_DIR = Path("Data/Processed")


def _model_is_trained() -> bool:
    """
    Return True when the trained model artifact exists.

    This check is used by sidebar actions so the module stays importable even
    before training has run.
    """
    return MODEL_PATH.exists()


def _current_circuit_rows() -> list[dict[str, Any]]:
    """
    Convert the current Streamlit session circuit components into flat rows.

    Each row represents:
    (component, pin) -> net
    """
    graph = st.session_state.get("circuit_graph")

    if graph is not None:
        rows = graph.to_rows()
        if rows:
            return rows

    rows: list[dict[str, Any]] = []

    circuit_name = st.session_state.get("circuit_name", "")
    circuit_description = st.session_state.get("circuit_description", "")
    circuit_notes = st.session_state.get("circuit_notes", "")

    for component in st.session_state.get("circuit_components", []):
        for pin in component.pin_assignments:
            rows.append(
                {
                    "circuit_name": circuit_name,
                    "description": circuit_description,
                    "ref_des": component.ref_des,
                    "component_kind": component.component_kind,
                    "component_type": component.component_type,
                    "component_value": component.component_value,
                    "component_value_type": component.component_value_type,
                    "pin_name": pin.pin_name,
                    "pin_role": pin.pin_role,
                    "net_name": pin.net_name,
                    "component_notes": component.component_notes,
                    "circuit_notes": circuit_notes,
                }
            )

    return rows


def _prediction_display_payload(result: dict[str, Any]) -> dict[str, Any]:
    """
    Build the sidebar-friendly prediction summary.
    """
    probabilities = result.get("probabilities") or {}
    confidence = result.get("confidence")

    if confidence is None and probabilities:
        confidence = max(probabilities.values())

    return {
        "prediction": result.get("prediction", "UNKNOWN"),
        "confidence": confidence,
        "probabilities": probabilities,
        "input_row": result.get("input_row", {}),
        "raw_prediction": result,
    }


def render_prediction_summary(prediction: dict[str, Any]) -> None:
    """
    Render the compact prediction summary requested by the UI.
    """
    confidence = prediction.get("confidence")
    probabilities = prediction.get("probabilities") or {}

    st.sidebar.markdown("**Predicted:**")
    st.sidebar.write(prediction.get("prediction", "UNKNOWN"))

    if confidence is not None:
        st.sidebar.write(f"Confidence: {confidence:.0%}")

    if probabilities:
        top_predictions = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        st.sidebar.write("Top probabilities:")
        for label, probability in top_predictions:
            st.sidebar.write(f"{label}: {probability:.0%}")


def train_model_from_sidebar() -> None:
    """
    Train the current ML model from the sidebar.

    Calls src.train.train_model() and saves artifacts.
    """
    try:
        from src.train import save_artifacts, train_model

        with st.spinner("Training model..."):
            result = train_model()
            save_artifacts(result)

        st.sidebar.success("Training complete. Model artifacts saved.")

    except FileNotFoundError as error:
        st.sidebar.error(f"Training data or file missing: {error}")

    except Exception as error:
        st.sidebar.error(f"Training failed: {type(error).__name__}: {error}")


def predict_from_sidebar():
    """
    Run a one-row prediction from the most recent circuit row in session state.

    Workflow role:
        Lightweight sidebar action for quick prediction feedback during circuit
        editing in the Streamlit UI.
    """
    if not _model_is_trained():
        st.sidebar.error("Model not trained yet. Click 'Train Model' first.")
        return

    rows = _current_circuit_rows()

    if not rows:
        st.sidebar.warning("Add components first.")
        return

    try:
        from src import predict_record

        latest = rows[-1]

        with st.spinner("Predicting..."):
            result = predict_record(latest)

        prediction = _prediction_display_payload(result)
        st.session_state.latest_prediction = prediction

        st.sidebar.success("Prediction complete")
        render_prediction_summary(prediction)

    except Exception as e:
        st.sidebar.error(f"Prediction failed: {e}")


def save_current_circuit_export() -> None:
    """
    Save the current in-memory circuit rows to a CSV export.

    This is the MVP save path until src.db.mutations is connected.
    """
    rows = _current_circuit_rows()

    if not rows:
        st.sidebar.warning("No circuit rows available to save.")
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    circuit_name = st.session_state.get("circuit_name", "unnamed_circuit")
    safe_name = circuit_name.strip().lower().replace(" ", "_") or "unnamed_circuit"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    export_path = EXPORT_DIR / f"{safe_name}_{timestamp}.csv"

    df = pd.DataFrame(rows)
    df.to_csv(export_path, index=False)

    st.session_state.latest_export_path = str(export_path)

    st.sidebar.success("Circuit exported.")
    st.sidebar.write(str(export_path))


def save_current_circuit_to_db() -> None:
    """
    Future DB save function.

    This placeholder is separated from CSV export so the sidebar already has
    the correct function boundary for src.db.mutations integration.
    """
    try:
        from src.db.mutations import save_circuit_from_rows

        rows = _current_circuit_rows()

        if not rows:
            st.sidebar.warning("No circuit rows available to save.")
            return

        with st.spinner("Saving circuit to database..."):
            save_circuit_from_rows(rows)

        st.sidebar.success("Circuit saved to database.")

    except ModuleNotFoundError:
        st.sidebar.info("Database save is not connected yet. Use CSV export for MVP.")

    except ImportError:
        st.sidebar.info("Database save function is not implemented yet.")

    except Exception as error:
        st.sidebar.error(f"Database save failed: {type(error).__name__}: {error}")


def render_ml_actions() -> None:
    """
    Render train, predict, and save/export sidebar actions.
    """
    st.sidebar.markdown("### ML Actions")

    if st.sidebar.button("Train Model"):
        train_model_from_sidebar()

    if not _model_is_trained():
        st.sidebar.caption("Train the model before running prediction.")

    if st.sidebar.button("Predict Next Component"):
        predict_from_sidebar()

    st.sidebar.markdown("### Save")

    if st.sidebar.button("Export Circuit CSV"):
        save_current_circuit_export()

    if st.sidebar.button("Save Circuit to DB"):
        save_current_circuit_to_db()


def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar controls and return selected UI state.
    """
    st.sidebar.title("Yggdrasil")

    selected_mode = st.sidebar.radio(
        "Add",
        list(ADD_OPTIONS.keys()),
        help="Choose what type of item to create or edit.",
    )

    graph_view = st.sidebar.radio(
        "Graph View",
        ["Concept", "Technical"],
        help="Concept hides detail; Technical shows ML/debug metadata.",
    )

    st.sidebar.markdown("### Help")
    st.sidebar.info(ADD_OPTIONS[selected_mode])

    st.sidebar.markdown("### MVP Scope")
    st.sidebar.write(
        "- Create circuits\n"
        "- Add components\n"
        "- Assign pins to nets\n"
        "- Prepare data for ML"
    )

    render_ml_actions()

    if "graph_validation" in st.session_state:
        validation = st.session_state.graph_validation
        st.sidebar.markdown("### Graph State")
        st.sidebar.write(validation.get("graph_state", ["UNKNOWN"])[0])

        warning_count = len(validation.get("warnings", []))
        error_count = len(validation.get("errors", []))
        st.sidebar.caption(f"{warning_count} warning(s), {error_count} error(s)")

    if "latest_prediction" in st.session_state:
        st.sidebar.markdown("### Latest Prediction")
        render_prediction_summary(st.session_state.latest_prediction)

    if "latest_export_path" in st.session_state:
        st.sidebar.markdown("### Latest Export")
        st.sidebar.code(st.session_state.latest_export_path)

    return {
        "selected_mode": selected_mode,
        "graph_view": graph_view,
    }

