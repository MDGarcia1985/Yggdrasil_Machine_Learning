"""
Edges/connections for the circuit graph.

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

Circuit rule:
Wires do not connect node-to-node directly.
Wires connect component pins to named nets.

Shape:
(ref_des, pin_name, net_name)
"""

from typing import List, Tuple


def init_default_edges() -> List[Tuple[str, str, str]]:
    """
    Purpose:
        Supply canonical pin→net wiring parallel to `init_default_nodes`.

    Design:
        Tuple order `(ref_des, pin_name, net_name)` matches `graph.PinNetEdge`.

    Workflow:
        Loaded alongside default nodes when the UI opens the sample design.

    Data handoff:
        Assigned to `CircuitGraph.edges`; combined with nodes in `to_rows()` for ML.
    """

    return [
        # Power source
        ("V1", "POS", "NET_VIN"),
        ("V1", "NEG", "NET_GND"),

        # RC input / timing network
        ("R1", "A", "NET_VIN"),
        ("R1", "B", "NET_SIGNAL_NODE"),

        ("C1", "A", "NET_SIGNAL_NODE"),
        ("C1", "B", "NET_GND"),

        # Schmitt trigger block
        ("U1", "VCC", "NET_VIN"),
        ("U1", "GND", "NET_GND"),
        ("U1", "IN", "NET_SIGNAL_NODE"),
        ("U1", "OUT", "NET_DRIVER_IN"),

        # Transistor switching stage
        ("Q1", "BASE", "NET_DRIVER_IN"),
        ("Q1", "EMITTER", "NET_GND"),
        ("Q1", "COLLECTOR", "NET_LOAD_LOW"),

        # Load/output stage
        ("LOAD1", "A", "NET_VIN"),
        ("LOAD1", "B", "NET_LOAD_LOW"),
    ]
