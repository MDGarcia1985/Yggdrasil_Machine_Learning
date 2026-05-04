"""
Graph container that ties circuit nodes + pin-to-net edges together.

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

NetworkX / GNN ready.

Circuit rule:
A connection is not node-to-node.
A connection is component pin -> net.
"""

from typing import Dict, List, Tuple, Optional, Any
from .nodes import Node, init_default_nodes
from .edges import init_default_edges


PinNetEdge = Tuple[str, str, str]
# Shape:
# (ref_des, pin_name, net_name)


class CircuitGraph:
    """
    Central graph model for the circuit ML system.

    Owns:
    - circuit metadata
    - component nodes
    - pin-to-net connections

    Exports:
    - flattened ML/database rows
    """

    def __init__(
        self,
        circuit_name: str = "default_circuit",
        description: str = "",
        notes: str = "",
    ):
        self.circuit_name = circuit_name
        self.description = description
        self.notes = notes

        self.nodes: Dict[str, Node] = {}
        self.edges: List[PinNetEdge] = []

    def load_default(self):
        """
        Load sample circuit graph.
        """
        self.nodes = init_default_nodes()
        self.edges = init_default_edges()

    def reset(self):
        """
        Reset graph back to the default sample circuit.
        """
        self.load_default()

    def add_node(self, node: Node):
        """
        Add or replace a circuit component node.
        """
        node.ref_des = node.ref_des.strip().upper()
        self.nodes[node.ref_des] = node

    def add_edge(self, ref_des: str, pin_name: str, net_name: str):
        """
        Add a pin-to-net connection.
        """
        self.edges.append(
            (
                ref_des.strip().upper(),
                pin_name.strip().upper(),
                normalize_net_name(net_name),
            )
        )

    def get_node(self, ref_des: str) -> Optional[Node]:
        """
        Get component node by reference designator.
        """
        return self.nodes.get(ref_des.strip().upper())

    def get_nodes_list(self) -> List[Node]:
        return list(self.nodes.values())

    def get_edges_list(self) -> List[PinNetEdge]:
        return self.edges

    def get_nets(self) -> List[str]:
        """
        Return unique net names used in the graph.
        """
        return sorted({net_name for _, _, net_name in self.edges})

    def get_edges_for_node(self, ref_des: str) -> List[PinNetEdge]:
        """
        Return all pin-to-net edges for one component.
        """
        normalized_ref = ref_des.strip().upper()

        return [
            edge for edge in self.edges
            if edge[0] == normalized_ref
        ]

    def get_edges_for_net(self, net_name: str) -> List[PinNetEdge]:
        """
        Return all component pins connected to a net.
        """
        normalized = normalize_net_name(net_name)

        return [
            edge for edge in self.edges
            if edge[2] == normalized
        ]

    def to_rows(self) -> List[Dict[str, Any]]:
        """
        Flatten graph into the current ML/database row format.

        Output shape matches:
        circuit_name
        description
        ref_des
        component_kind
        component_type
        component_value
        component_value_type
        pin_name
        net_name
        circuit_notes
        """

        rows: List[Dict[str, Any]] = []

        for ref_des, pin_name, net_name in self.edges:
            node = self.get_node(ref_des)

            if node is None:
                continue

            rows.append(
                {
                    "circuit_name": self.circuit_name,
                    "description": self.description,
                    "ref_des": node.ref_des,
                    "component_kind": node.component_kind,
                    "component_type": node.component_type,
                    "component_value": node.value,
                    "component_value_type": node.value_type,
                    "pin_name": pin_name,
                    "net_name": net_name,
                    "circuit_notes": self.notes,
                    "component_notes": node.notes,
                }
            )

        return rows


def normalize_net_name(raw_name: str) -> str:
    """
    Normalize net names for consistent database and ML usage.

    User can type:
    VIN

    System stores:
    NET_VIN
    """
    cleaned = raw_name.strip().upper().replace(" ", "_")

    if cleaned.startswith("NET_"):
        return cleaned

    if cleaned.startswith("NET-"):
        return cleaned.replace("NET-", "NET_")

    return f"NET_{cleaned}"
