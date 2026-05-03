"""
Edges/connections for the circuit graph.

Circuit rule:
Wires do not connect node-to-node directly.
Wires connect component pins to named nets.

Shape:
(ref_des, pin_name, net_name)
"""

from typing import List, Tuple


def init_default_edges() -> List[Tuple[str, str, str]]:
    """
    Sample pin-to-net connections for the default circuit graph.

    Represents:
    V1 -> RC input conditioning -> Schmitt trigger -> transistor/load stage
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
