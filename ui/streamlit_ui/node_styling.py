"""
Node styling functions (color, shape, label logic) for circuit graph UI.

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247
"""

from src.nodes import Node


def get_node_style(n: Node, view: str = "Concept") -> dict:
    """
    Return style dict for rendering nodes.
    Encodes circuit state and functional role.
    """

    # State-driven styling
    if n.state == "INVALID":
        return {"color": "#8B0000", "fontcolor": "red", "fillcolor": "#FFE0E0"}

    if n.state == "COMPLETE":
        return {"color": "#228B22", "fillcolor": "#90EE90"}  # green

    if n.state == "PARTIAL":
        return {"color": "#FFD700", "fillcolor": "#FFFACD"}  # yellow

    if n.state == "VALIDATED":
        return {"color": "#1E90FF", "fillcolor": "#ADD8E6"}  # blue

    # Default / draft
    return {"color": "#808080", "fillcolor": "#D3D3D3"}


def get_node_label(n: Node, view: str = "Concept") -> str:
    """
    Concept vs Technical view labels.

    Concept:
        Simplified functional meaning

    Technical:
        Full ML/EDA detail
    """

    if view == "Concept":
        return _concept_label(n)

    return _technical_label(n)


# -------------------------
# Label helpers
# -------------------------

def _concept_label(n: Node) -> str:
    """
    Human-friendly simplified label.
    """

    # Example:
    # R1
    # resistor
    # filter

    role = n.role if n.role != "UNASSIGNED" else ""

    return f"{n.ref_des}\n{n.component_type}\n{role}".strip()


def _technical_label(n: Node) -> str:
    """
    Detailed label for engineering / ML debugging.
    """

    # Example:
    # R1 (resistor)
    # value: 10000 ohm
    # role: FILTER
    # state: COMPLETE

    value_str = ""
    if n.value:
        value_str = f"value: {n.value} {n.value_type}".strip()

    return (
        f"{n.ref_des} ({n.component_type})\n"
        f"{value_str}\n"
        f"role: {n.role}\n"
        f"state: {n.state}"
    )

