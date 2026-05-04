"""
Edge styling for circuit graph (wires / nets).

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247
"""

from src.graph import CircuitGraph


def get_edge_style(ref_des: str, pin_name: str, net_name: str, graph: CircuitGraph) -> dict:
    """
    Return style for a pin-to-net connection.

    Styling reflects:
    - net type (power, ground, signal)
    - connectivity clarity
    """

    net = net_name.upper()

    # Net type coloring
    if "GND" in net or "GROUND" in net:
        color = "#8B4513"  # brown
    elif "VIN" in net or "VCC" in net or "VDD" in net:
        color = "#FF4500"  # orange/red (power)
    else:
        color = "#228B22"  # green (signal)

    # Connection density (fanout)
    connections = graph.get_edges_for_net(net_name)

    width = 3
    if len(connections) > 4:
        width = 4  # thicker for shared nets (bus-like behavior)
    elif len(connections) == 1:
        color = "#808080"  # floating net warning

    return {
        "width": width,
        "color": color,
        "label": net_name,
    }
