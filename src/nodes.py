"""
This graph editor is inspired by a project I did for tinyCore, an open source ESP32 based project that seeks to make learning embedded systems more intuitive and fun. I created tinyTrainer to demonstrate how the tinyCore can be used for PLC training.

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247
"""

from dataclasses import dataclass, field
from typing import Dict, Any

ROLES = [
    "UNASSIGNED",

    # Signal roles
    "INPUT",
    "OUTPUT",
    "BIDIRECTIONAL",

    # Power roles
    "POWER",
    "GROUND",
    "REFERENCE",

    # Functional behavior
    "PASSIVE",
    "ACTIVE",
    "CONTROL",

    # Circuit intent (ML useful)
    "FILTER",
    "AMPLIFICATION",
    "SWITCHING",
    "TIMING",
    "PROTECTION",
    "FEEDBACK",
    "DRIVER",
]


STATES = [
    "UNASSIGNED",

    # Design phase
    "DRAFT",
    "PARTIAL",
    "COMPLETE",

    # Validation phase
    "VALIDATED",
    "INVALID",

    # Runtime / simulation (future)
    "ACTIVE",
    "INACTIVE",
    "FAULT",
]


BEHAVIOR = [
    "LINEAR",
    "NONLINEAR",
    "DIGITAL",
    "MIXED_SIGNAL",
    "POWER",
    "RF",
]


from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Pin:
    name: str
    role: str = "UNASSIGNED"   # INPUT, OUTPUT, POWER, GROUND, PASSIVE
    net: str = ""              # NET_VIN, NET_GND, etc.


@dataclass
class Node:
    """
    Circuit graph node representing a component.
    """

    # Identity
    ref_des: str                 # R1, C1, U1
    component_type: str          # resistor, capacitor, op_amp, etc.
    component_kind: str          # primitive, structure, block, part

    # Electrical meaning
    value: str = ""              # 10k, 0.1uF, NE555
    value_type: str = "none"     # ohm, uF, V, etc.

    # Behavior / ML semantics
    role: str = "UNASSIGNED"     # FILTER, AMPLIFICATION, etc.
    state: str = "DRAFT"         # DRAFT, COMPLETE, VALIDATED

    # Connectivity
    pins: List[Pin] = field(default_factory=list)

    # UI position (keep this — useful)
    x: float = 0.0
    y: float = 0.0

    # Free-form metadata
    notes: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


def init_default_nodes() -> Dict[str, Node]:
    """
    Sample circuit nodes for ML and UI testing.
    Represents a simple signal conditioning + output stage.
    """

    return {
        "V1": Node(
            ref_des="V1",
            component_type="voltage_source",
            component_kind="source",
            role="POWER",
            state="ACTIVE",
            value="5",
            value_type="V",
            x=150,
            y=300,
        ),

        "R1": Node(
            ref_des="R1",
            component_type="resistor",
            component_kind="primitive",
            role="FILTER",
            state="COMPLETE",
            value="10000",
            value_type="ohm",
            x=350,
            y=200,
        ),

        "C1": Node(
            ref_des="C1",
            component_type="capacitor",
            component_kind="primitive",
            role="TIMING",
            state="COMPLETE",
            value="0.1",
            value_type="uF",
            x=350,
            y=400,
        ),

        "U1": Node(
            ref_des="U1",
            component_type="schmitt_trigger",
            component_kind="block",
            role="CONTROL",
            state="COMPLETE",
            x=550,
            y=300,
        ),

        "Q1": Node(
            ref_des="Q1",
            component_type="bjt_npn",
            component_kind="primitive",
            role="SWITCHING",
            state="COMPLETE",
            x=750,
            y=300,
        ),

        "LOAD1": Node(
            ref_des="LOAD1",
            component_type="load",
            component_kind="structure",
            role="OUTPUT",
            state="COMPLETE",
            x=950,
            y=300,
        ),
    }