"""
Circuit graph validation / lightweight simulation helpers.

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

This is NOT SPICE.

Purpose:
- mark nodes/nets as draft, complete, or invalid
- detect obvious circuit-entry problems
- support UI feedback before SQL/ML handoff
- prepare future rule-based validation and prediction hints
"""

from typing import Dict, List, Tuple
from .nodes import Node
from .graph import CircuitGraph, normalize_net_name


def tick_sim(graph: CircuitGraph) -> Dict[str, List[str]]:
    """
    Purpose:
        Aggregate lightweight connectivity checks into UI-friendly messages.

    Design:
        Mutates no graph topology; only reads `CircuitGraph` and accumulates
        strings plus a coarse `graph_state` label.

    Workflow:
        Called on timer or after edits in the Streamlit graph UI.

    Data handoff:
        Inputs: live `CircuitGraph` instance.
        Outputs: dict with keys `graph_state`, `warnings`, `errors` (string lists).
    """

    warnings: List[str] = []
    errors: List[str] = []

    validate_missing_connections(graph, warnings, errors)
    validate_power_and_ground(graph, warnings, errors)
    validate_floating_nets(graph, warnings, errors)

    if errors:
        graph_state = "INVALID"
    elif warnings:
        graph_state = "PARTIAL"
    else:
        graph_state = "COMPLETE"

    return {
        "graph_state": [graph_state],
        "warnings": warnings,
        "errors": errors,
    }


def validate_missing_connections(
    graph: CircuitGraph,
    warnings: List[str],
    errors: List[str],
) -> None:
    """
    Purpose:
        Warn when a placed component has no edges (floating part).

    Design:
        Builds a set of refs that appear on any edge, then subtracts from all
        `graph.nodes` keys.

    Workflow:
        First validator inside `tick_sim`.

    Data handoff:
        Mutates `warnings` in place (append-only); ignores `errors` today.
    """

    connected_refs = {ref_des for ref_des, _, _ in graph.edges}

    for ref_des, node in graph.nodes.items():
        if ref_des not in connected_refs:
            warnings.append(f"{ref_des} has no net connections.")


def validate_power_and_ground(
    graph: CircuitGraph,
    warnings: List[str],
    errors: List[str],
) -> None:
    """
    Purpose:
        Heuristically detect missing supply rails based on net name substrings.

    Design:
        Simple keyword scan (not a full BOM/netclass analysis); false positives
        are acceptable for MVP hints.

    Workflow:
        Second validator inside `tick_sim`.

    Data handoff:
        Reads `graph.get_nets()`; appends human-readable strings to `warnings`.
    """

    nets = graph.get_nets()

    has_power = any(
        "VIN" in net or "VCC" in net or "VDD" in net or "POWER" in net
        for net in nets
    )

    has_ground = any(
        "GND" in net or "GROUND" in net or "VSS" in net
        for net in nets
    )

    if not has_power:
        warnings.append("No obvious power net found.")

    if not has_ground:
        warnings.append("No obvious ground net found.")


def validate_floating_nets(
    graph: CircuitGraph,
    warnings: List[str],
    errors: List[str],
) -> None:
    """
    Purpose:
        Flag nets that only connect to a single pin (cannot carry current between parts).

    Design:
        Uses `get_edges_for_net` length check; informational warning only.

    Workflow:
        Third validator inside `tick_sim`.

    Data handoff:
        Appends to `warnings` with contextual ref/pin names for debugging.
    """

    for net in graph.get_nets():
        connected_edges = graph.get_edges_for_net(net)

        if len(connected_edges) == 1:
            ref_des, pin_name, _ = connected_edges[0]
            warnings.append(
                f"{net} only connects to {ref_des}.{pin_name}; possible floating net."
            )


def set_role(graph: CircuitGraph, ref_des: str, role: str) -> None:
    """
    Purpose:
        Annotate a component with a coarse behavioral role for UI and future rules.

    Design:
        Updates `node.role` and nudges `node.state` out of `UNASSIGNED`/`DRAFT`
        when a concrete role is chosen.

    Workflow:
        Role picker widgets in the editor.

    Data handoff:
        Mutates `Node` in place inside `graph.nodes`; no return value.
    """

    node = graph.get_node(ref_des)

    if node is None:
        return

    node.role = role

    if role == "UNASSIGNED":
        node.state = "UNASSIGNED"
    elif node.state in ("UNASSIGNED", "DRAFT"):
        node.state = "PARTIAL"


def mark_node_complete(graph: CircuitGraph, ref_des: str) -> None:
    """
    Purpose:
        Transition a node's `state` to `COMPLETE` when all declared pins are wired.

    Design:
        Compares `node.pins` names against pins present on outgoing edges for
        the same `ref_des`; if pins list empty, never marks complete.

    Workflow:
        After edge edits or auto-router completion hooks.

    Data handoff:
        Reads `graph.edges` and mutates matching `Node.state`.
    """

    node = graph.get_node(ref_des)

    if node is None:
        return

    connected_pins = {
        pin_name
        for edge_ref, pin_name, _ in graph.edges
        if edge_ref == ref_des
    }

    node_pin_names = {pin.name for pin in node.pins}

    if node_pin_names and node_pin_names.issubset(connected_pins):
        node.state = "COMPLETE"
    else:
        node.state = "PARTIAL"


def mark_prediction_accepted(
    graph: CircuitGraph,
    ref_des: str,
    predicted_component_type: str,
) -> None:
    """
    Purpose:
        Persist user confirmation that the ML suggestion was correct.

    Design:
        Writes into `node.properties` bag and sets `state="VALIDATED"`; no
        training loop is triggered here (future data pipeline hook).

    Workflow:
        UI "accept prediction" button handler.

    Data handoff:
        Mutates `Node.properties` for later export to DB or labeling jobs.
    """

    node = graph.get_node(ref_des)

    if node is None:
        return

    node.properties["prediction_accepted"] = True
    node.properties["predicted_component_type"] = predicted_component_type
    node.state = "VALIDATED"


def mark_prediction_rejected(
    graph: CircuitGraph,
    ref_des: str,
    predicted_component_type: str,
) -> None:
    """
    Purpose:
        Record that the user disagreed with the model's suggested component type.

    Design:
        Sets `prediction_accepted=False` and moves node to `REVIEW` state for
        visibility in UI queues.

    Workflow:
        UI "reject prediction" path paired with manual correction flows.

    Data handoff:
        Same `node.properties` contract as `mark_prediction_accepted`.
    """

    node = graph.get_node(ref_des)

    if node is None:
        return

    node.properties["prediction_accepted"] = False
    node.properties["predicted_component_type"] = predicted_component_type
    node.state = "REVIEW"
