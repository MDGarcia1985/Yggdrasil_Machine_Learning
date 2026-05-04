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
    Run one lightweight validation pass over the circuit graph.

    Returns:
        Dictionary of warnings/errors for UI display.
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
    Check for nodes without pin-to-net assignments.
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
    Check that the circuit has at least one power net and one ground net.
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
    Warn when a net has only one connected pin.

    Single-pin nets may be intentional test points, but they are often mistakes.
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
    Set functional/electrical role for a component node.
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
    Mark a component as complete after required pins are assigned.
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
    Mark a prediction as accepted for future validation/reinforcement logging.

    This does not retrain the model directly.
    It only records user validation intent on the node.
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
    Mark a prediction as rejected for future validation/reinforcement logging.
    """

    node = graph.get_node(ref_des)

    if node is None:
        return

    node.properties["prediction_accepted"] = False
    node.properties["predicted_component_type"] = predicted_component_type
    node.state = "REVIEW"
