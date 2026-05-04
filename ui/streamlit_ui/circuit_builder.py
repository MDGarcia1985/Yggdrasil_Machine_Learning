"""
circuit_builder.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

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

from dataclasses import dataclass
from html import escape
from typing import List

import streamlit as st

from src.graph import CircuitGraph, normalize_net_name
from src.nodes import Node, Pin
from src.simulator import tick_sim
from ui.streamlit_ui.edge_styling import get_edge_style
from ui.streamlit_ui.graph_styling import get_graph_config
from ui.streamlit_ui.node_styling import get_node_label, get_node_style


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

SOURCE_VALUE_TYPES = {
    "voltage": "V",
    "current": "A",
    "signal": "Hz",
}

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
    normalized_kind = component_kind.strip().lower()
    normalized_type = component_type.strip().lower()

    if component_kind == "Custom":
        return [f"PIN_{i}" for i in range(1, custom_pin_count + 1)]

    if normalized_kind == "source":
        return DEFAULT_PINS[normalized_type]

    return DEFAULT_PINS.get(normalized_type, ["A", "B"])


def _get_value_type_options(component_kind: str, component_type: str) -> tuple[List[str], int]:
    """
    Return value-type options for the selected component.
    """
    normalized_kind = component_kind.strip().lower()
    normalized_type = component_type.strip().lower()

    if normalized_kind == "source":
        source_value_type = SOURCE_VALUE_TYPES[normalized_type]
        return [source_value_type], 0

    return VALUE_TYPES, 0


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


def _build_circuit_graph_from_session() -> CircuitGraph:
    """
    Convert the current form/session data into the shared CircuitGraph model.
    """
    graph = CircuitGraph(
        circuit_name=st.session_state.get("circuit_name", ""),
        description=st.session_state.get("circuit_description", ""),
        notes=st.session_state.get("circuit_notes", ""),
    )

    for component in st.session_state.get("circuit_components", []):
        pins = [
            Pin(
                name=pin.pin_name,
                role=pin.pin_role.upper(),
                net=normalize_net_name(pin.net_name),
            )
            for pin in component.pin_assignments
        ]

        graph.add_node(
            Node(
                ref_des=component.ref_des,
                component_type=component.component_type,
                component_kind=component.component_kind,
                value=component.component_value,
                value_type=component.component_value_type,
                pins=pins,
                notes=component.component_notes,
            )
        )

        for pin in component.pin_assignments:
            graph.add_edge(component.ref_des, pin.pin_name, pin.net_name)

    return graph


def _dot_attrs(attrs: dict[str, object]) -> str:
    """
    Format Graphviz attributes safely.
    """
    return ", ".join(
        f'{key}="{escape(str(value), quote=True)}"'
        for key, value in attrs.items()
        if value is not None
    )


def _render_circuit_graph(graph: CircuitGraph, view: str) -> None:
    """
    Render the circuit graph using the styling helpers.
    """
    if not graph.nodes:
        st.info("Add components to see the circuit graph.")
        return

    config = get_graph_config(view)
    dot_lines = [
        "digraph CircuitGraph {",
        "rankdir=LR;",
        f'bgcolor="{config["background"]}";',
        "graph [pad=0.4, nodesep=0.8, ranksep=1.0];",
        'node [shape=box, style="rounded,filled", fontname="Consolas"];',
        'edge [fontname="Consolas"];',
    ]

    for node in graph.get_nodes_list():
        style = get_node_style(node, view)
        attrs = {
            "label": get_node_label(node, view),
            "color": style.get("color"),
            "fillcolor": style.get("fillcolor"),
            "fontcolor": style.get("fontcolor", "#111111"),
            "penwidth": 2,
        }
        dot_lines.append(f'"{node.ref_des}" [{_dot_attrs(attrs)}];')

    for net in graph.get_nets():
        attrs = {
            "label": net,
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "#1F2933",
            "fontcolor": "#E5E7EB",
            "color": "#4B5563",
        }
        dot_lines.append(f'"{net}" [{_dot_attrs(attrs)}];')

    for ref_des, pin_name, net_name in graph.get_edges_list():
        style = get_edge_style(ref_des, pin_name, net_name, graph)
        attrs = {
            "label": pin_name,
            "color": style.get("color"),
            "penwidth": style.get("width"),
            "fontcolor": style.get("color"),
        }
        dot_lines.append(f'"{ref_des}" -> "{net_name}" [{_dot_attrs(attrs)}];')

    dot_lines.append("}")

    st.graphviz_chart("\n".join(dot_lines), use_container_width=True)


def _render_validation_status(graph: CircuitGraph) -> None:
    """
    Show lightweight simulator/validator output.
    """
    sim_result = tick_sim(graph)
    st.session_state.graph_validation = sim_result

    graph_state = sim_result["graph_state"][0]

    if graph_state == "COMPLETE":
        st.success("Graph state: COMPLETE")
    elif graph_state == "PARTIAL":
        st.warning("Graph state: PARTIAL")
    else:
        st.error("Graph state: INVALID")

    if sim_result["warnings"]:
        with st.expander("Validation warnings", expanded=False):
            for warning in sim_result["warnings"]:
                st.write(f"- {warning}")

    if sim_result["errors"]:
        with st.expander("Validation errors", expanded=True):
            for error in sim_result["errors"]:
                st.write(f"- {error}")


def render_circuit_builder(graph_view: str = "Concept") -> None:
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

    value_type_options, value_type_index = _get_value_type_options(
        component_kind,
        component_type,
    )
    component_value_type = st.selectbox(
        "Component Value Type",
        value_type_options,
        index=value_type_index,
    )

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

    graph = _build_circuit_graph_from_session()
    st.session_state.circuit_graph = graph

    st.divider()

    st.subheader("Circuit Graph")
    _render_validation_status(graph)
    _render_circuit_graph(graph, graph_view)

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
