"""
circuit_builder.py

Purpose:
Streamlit circuit-builder UI for entering circuit-level data and component
pin-to-net assignments.

Design:
- The user creates a circuit.
- The user adds one component row at a time.
- Known components use database-driven pins later.
- Custom components expose component_pin_count and dynamic pin rows.
- For MVP, local session_state stores the circuit until DB mutations are added.

Future integration:
- src.db.queries      -> dropdown options
- src.db.mutations    -> save circuit/components/connections
- src.predict         -> prediction after each component entry
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import streamlit as st


COMPONENT_KINDS = [
    "Primitive",
    "Structure",
    "Functional Block",
    "Part",
    "Source",
    "Custom",
]

MOCK_COMPONENT_TYPES = {
    "Primitive": [
        "resistor",
        "capacitor",
        "diode",
        "transistor",
        "switch",
        "inductor",
        "fuse",
    ],
    "Structure": [
        "voltage_divider",
        "rc_low_pass",
        "rc_high_pass",
        "pull_up",
        "pull_down",
        "transistor_switch",
    ],
    "Functional Block": [
        "555_timer",
        "schmitt_trigger",
        "comparator",
        "op_amp",
        "motor_driver",
        "mcu",
        "fpga",
    ],
    "Part": [
        "NE555",
        "LM358",
        "BD64950EFJ-E2",
        "BC549",
        "ESP32",
    ],
    "Source": [
        "voltage",
        "current",
        "signal",
    ],
    "Custom": [
        "custom_component",
    ],
}

VALUE_TYPES = [
    "none",
    "ohm",
    "uF",
    "nF",
    "pF",
    "V",
    "A",
    "Hz",
    "part_number",
]

PIN_ROLE_OPTIONS = [
    "unknown",
    "input",
    "output",
    "passive",
    "power",
    "ground",
    "control",
    "bidirectional",
]


DEFAULT_PINS = {
    "resistor": ["A", "B"],
    "capacitor": ["A", "B"],
    "diode": ["ANODE", "CATHODE"],
    "transistor": ["COLLECTOR", "BASE", "EMITTER"],
    "switch": ["A", "B"],
    "inductor": ["A", "B"],
    "fuse": ["A", "B"],
    "voltage": ["POS", "NEG"],
    "current": ["POS", "NEG"],
    "signal": ["OUT", "GND"],
    "555_timer": ["GND", "TRIG", "OUT", "RESET", "CTRL", "THRESH", "DISCH", "VCC"],
    "schmitt_trigger": ["IN", "OUT", "VCC", "GND"],
    "comparator": ["IN+", "IN-", "OUT", "VCC", "GND"],
    "op_amp": ["IN+", "IN-", "OUT", "V+", "V-"],
    "motor_driver": ["IN1", "IN2", "OUT1", "OUT2", "VCC", "GND"],
    "mcu": ["VCC", "GND", "GPIO1", "GPIO2"],
    "fpga": ["VCC", "GND", "IO1", "IO2"],
}


REF_DES_PREFIX = {
    "resistor": "R",
    "capacitor": "C",
    "diode": "D",
    "transistor": "Q",
    "switch": "SW",
    "inductor": "L",
    "fuse": "F",
    "voltage": "V",
    "current": "I",
    "signal": "SIG",
    "555_timer": "U",
    "schmitt_trigger": "U",
    "comparator": "U",
    "op_amp": "U",
    "motor_driver": "U",
    "mcu": "U",
    "fpga": "U",
    "custom_component": "X",
}


@dataclass
class PinAssignment:
    pin_name: str
    pin_role: str
    net_name: str


@dataclass
class ComponentEntry:
    ref_des: str
    component_kind: str
    component_type: str
    component_value: str
    component_value_type: str
    component_notes: str
    pin_assignments: List[PinAssignment]


def _init_state() -> None:
    """
    Initialize Streamlit session state for circuit builder.
    """
    if "circuit_components" not in st.session_state:
        st.session_state.circuit_components = []

    if "ref_des_counts" not in st.session_state:
        st.session_state.ref_des_counts = {}


def _next_ref_des(component_type: str) -> str:
    """
    Auto-generate a reference designator from component type.
    """
    prefix = REF_DES_PREFIX.get(component_type, "X")
    current = st.session_state.ref_des_counts.get(prefix, 0) + 1
    st.session_state.ref_des_counts[prefix] = current

    return f"{prefix}{current}"


def _get_pins(component_kind: str, component_type: str, custom_pin_count: int) -> List[str]:
    """
    Return pins for known components or generated custom pin names.
    """
    if component_kind == "Custom":
        return [f"PIN_{i}" for i in range(1, custom_pin_count + 1)]

    return DEFAULT_PINS.get(component_type, ["A", "B"])


def _render_pin_assignments(pins: List[str]) -> List[PinAssignment]:
    """
    Render pin-to-net assignment rows.
    """
    assignments: List[PinAssignment] = []

    st.markdown("#### Pin-to-Net Assignments")

    for pin in pins:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            pin_name = st.text_input(
                f"Pin Name ({pin})",
                value=pin,
                key=f"pin_name_{pin}",
            )

        with col2:
            pin_role = st.selectbox(
                f"Pin Role ({pin})",
                PIN_ROLE_OPTIONS,
                key=f"pin_role_{pin}",
            )

        with col3:
            net_name = st.text_input(
                f"Net Name ({pin})",
                value="",
                placeholder="NET_VIN, NET_GND, NET_OUT...",
                key=f"net_name_{pin}",
            )

        assignments.append(
            PinAssignment(
                pin_name=pin_name.strip().upper(),
                pin_role=pin_role,
                net_name=net_name.strip().upper(),
            )
        )

    return assignments


def _render_current_components() -> None:
    """
    Display the current in-memory component list.
    """
    if not st.session_state.circuit_components:
        st.info("No components added yet.")
        return

    rows = []

    for component in st.session_state.circuit_components:
        for pin in component.pin_assignments:
            rows.append(
                {
                    "ref_des": component.ref_des,
                    "component_kind": component.component_kind,
                    "component_type": component.component_type,
                    "component_value": component.component_value,
                    "component_value_type": component.component_value_type,
                    "pin_name": pin.pin_name,
                    "pin_role": pin.pin_role,
                    "net_name": pin.net_name,
                    "component_notes": component.component_notes,
                }
            )

    st.dataframe(rows, use_container_width=True)


def render_circuit_builder() -> None:
    """
    Render the circuit builder UI.
    """
    _init_state()

    st.subheader("New Circuit")

    with st.form("circuit_header_form"):
        circuit_name = st.text_input("Circuit Name", placeholder="button_debounce")
        circuit_description = st.text_area(
            "Circuit Description",
            placeholder="Describe the purpose and behavior of this circuit.",
        )
        circuit_notes = st.text_area(
            "Circuit Notes",
            placeholder="Optional notes for documentation or ML validation.",
        )

        header_submitted = st.form_submit_button("Set Circuit Info")

    if header_submitted:
        st.session_state.circuit_name = circuit_name.strip()
        st.session_state.circuit_description = circuit_description.strip()
        st.session_state.circuit_notes = circuit_notes.strip()
        st.success("Circuit info set.")

    st.divider()

    st.subheader("Add Component")

    component_kind = st.selectbox("Component Kind", COMPONENT_KINDS)

    if component_kind == "Custom":
        st.info(
            "Thank you for choosing a future feature for creating custom component inputs. "
            "For MVP, custom components are allowed with user-defined pin counts, "
            "but the full Custom Component Builder is still under construction."
        )

    component_type_options = MOCK_COMPONENT_TYPES.get(component_kind, ["custom_component"])

    if component_kind == "Custom":
        component_type = st.text_input("Component Type", value="custom_component")
    else:
        component_type = st.selectbox("Component Type", component_type_options)

    component_value = st.text_input(
        "Component Value",
        placeholder="10000, 0.1, NE555, BD64950EFJ-E2...",
    )

    component_value_type = st.selectbox("Component Value Type", VALUE_TYPES)

    component_notes = st.text_area(
        "Component Notes",
        placeholder="Optional role, purpose, or design note.",
    )

    if component_kind == "Custom":
        custom_pin_count = st.number_input(
            "Component Pin Count",
            min_value=1,
            max_value=128,
            value=2,
            step=1,
        )
    else:
        custom_pin_count = 0

    pins = _get_pins(component_kind, component_type, int(custom_pin_count))
    pin_assignments = _render_pin_assignments(pins)

    if st.button("Add Component"):
        missing_nets = [p.pin_name for p in pin_assignments if not p.net_name]

        if missing_nets:
            st.error(f"Missing net assignments for pins: {missing_nets}")
            return

        ref_des = _next_ref_des(component_type)

        entry = ComponentEntry(
            ref_des=ref_des,
            component_kind=component_kind.lower().replace("functional block", "block"),
            component_type=component_type.strip().lower(),
            component_value=component_value.strip(),
            component_value_type=component_value_type,
            component_notes=component_notes.strip(),
            pin_assignments=pin_assignments,
        )

        st.session_state.circuit_components.append(entry)
        st.success(f"Added {ref_des}: {component_type}")

    st.divider()

    st.subheader("Current Circuit Rows")
    _render_current_components()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Components"):
            st.session_state.circuit_components = []
            st.session_state.ref_des_counts = {}
            st.warning("Cleared current component list.")

    with col2:
        if st.button("Save Circuit to DB"):
            st.info("Database write is not connected yet. This will call src.db.mutations later.")
