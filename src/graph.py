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
    Purpose:
        Hold an editable circuit as components (`Node`) plus pin→net edges.

    Design:
        Edges are `(ref_des, pin_name, net_name)` tuples, not node-to-node pairs,
        matching schematic connectivity and the ML CSV schema.

    Workflow:
        UI/graph editor mutates nodes/edges, then `to_rows()` feeds SQL/ML.

    Data handoff:
        `nodes` dict from `nodes.init_default_nodes` (or UI); `edges` list from
        `edges.init_default_edges` / user edits; `to_rows()` outputs dict rows
        compatible with `preprocess` / pandas.
    """

    def __init__(
        self,
        circuit_name: str = "default_circuit",
        description: str = "",
        notes: str = "",
    ):
        """
        Purpose:
            Create an empty graph shell ready for nodes and pin→net edges.

        Design:
            Stores metadata on `self` and initializes empty `nodes`/`edges`.

        Workflow:
            Construct then `load_default()` or programmatically `add_node`/`add_edge`.

        Data handoff:
            No external I/O; callers populate `self.nodes` and `self.edges`.
        """
        self.circuit_name = circuit_name
        self.description = description
        self.notes = notes

        self.nodes: Dict[str, Node] = {}
        self.edges: List[PinNetEdge] = []

    def load_default(self):
        """
        Purpose:
            Populate the graph with the built-in demo circuit (nodes + edges).

        Design:
            Delegates sample data to `init_default_nodes` / `init_default_edges`.

        Workflow:
            Called on fresh UI sessions or after `reset()`.

        Data handoff:
            Reads module defaults; writes into `self.nodes` and `self.edges`.
        """
        self.nodes = init_default_nodes()
        self.edges = init_default_edges()

    def reset(self):
        """
        Purpose:
            Restore the demo circuit, discarding unsaved user edits in memory.

        Design:
            Alias to `load_default()` for clearer intent in UI handlers.

        Workflow:
            "Reset canvas" actions.

        Data handoff:
            Replaces in-memory `nodes`/`edges` collections.
        """
        self.load_default()

    def add_node(self, node: Node):
        """
        Purpose:
            Register (or overwrite) a component keyed by reference designator.

        Design:
            Normalizes `ref_des` to uppercase trimmed form used as dict key.

        Workflow:
            UI drag-drop or import pipelines create `Node` then call this.

        Data handoff:
            Inputs: `Node` instance from `nodes` module or UI layer.
            Outputs: updated `self.nodes` map consumed by `to_rows` / simulator.
        """
        node.ref_des = node.ref_des.strip().upper()
        self.nodes[node.ref_des] = node

    def add_edge(self, ref_des: str, pin_name: str, net_name: str):
        """
        Purpose:
            Record that a component pin participates on a named net.

        Design:
            Appends to a list (allows duplicate checks elsewhere); net names run
            through `normalize_net_name` for storage consistency.

        Workflow:
            Wire tool in UI or bulk import.

        Data handoff:
            Inputs: raw strings from UI; outputs tuple on `self.edges` used by
            `simulator` validators and `to_rows`.
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
        Purpose:
            Fetch a `Node` by its ref_des, or `None` if unknown.

        Design:
            Case-insensitive lookup via normalized key.

        Workflow:
            Property panels, validation, `to_rows` join from edge to node fields.

        Data handoff:
            Inputs: ref string; outputs optional `Node` reference (not a copy).
        """
        return self.nodes.get(ref_des.strip().upper())

    def get_nodes_list(self) -> List[Node]:
        """
        Purpose:
            Expose all nodes as an ordered list for iteration/serialization.

        Design:
            Dict iteration order is insertion order (Python 3.7+); list is shallow.

        Workflow:
            UI component pickers, exports.

        Data handoff:
            Outputs new list wrapping existing `Node` objects.
        """
        return list(self.nodes.values())

    def get_edges_list(self) -> List[PinNetEdge]:
        """
        Purpose:
            Return the mutable edge list by reference for advanced callers.

        Design:
            No copy—mutations affect graph state immediately.

        Workflow:
            Rare; prefer `add_edge` unless bulk editing.

        Data handoff:
            Outputs `self.edges` list object.
        """
        return self.edges

    def get_nets(self) -> List[str]:
        """
        Purpose:
            List distinct nets currently used by any edge.

        Design:
            Set comprehension dedupes; `sorted` yields stable UI ordering.

        Workflow:
            Net-centric visualization and simulator power/ground checks.

        Data handoff:
            Outputs `List[str]` of net names derived from `self.edges`.
        """
        return sorted({net_name for _, _, net_name in self.edges})

    def get_edges_for_node(self, ref_des: str) -> List[PinNetEdge]:
        """
        Purpose:
            Filter edges to those originating from one component ref_des.

        Design:
            Linear scan over `self.edges` (small graphs; simple code).

        Workflow:
            Pin assignment UIs, `mark_node_complete` in `simulator`.

        Data handoff:
            Inputs: ref string; outputs list of `PinNetEdge` tuples.
        """
        normalized_ref = ref_des.strip().upper()

        return [
            edge for edge in self.edges
            if edge[0] == normalized_ref
        ]

    def get_edges_for_net(self, net_name: str) -> List[PinNetEdge]:
        """
        Purpose:
            List every `(ref_des, pin, net)` tuple attached to a given net.

        Design:
            Normalizes the query net the same way edges are stored.

        Workflow:
            Net highlight in UI, floating-net warnings in `simulator`.

        Data handoff:
            Inputs: raw or normalized net string; outputs filtered edge list.
        """
        normalized = normalize_net_name(net_name)

        return [
            edge for edge in self.edges
            if edge[2] == normalized
        ]

    def to_rows(self) -> List[Dict[str, Any]]:
        """
        Purpose:
            Serialize the graph into one dict per `(component pin → net)` edge.

        Design:
            Joins edge tuple with `Node` fields; skips orphan edges with missing
            nodes to avoid partial rows.

        Workflow:
            Export to CSV/SQL or `pd.DataFrame(rows)` for ML scoring.

        Data handoff:
            Outputs list of dicts with keys aligned to `preprocess.REQUIRED_COLUMNS`
            plus optional notes fields for richer storage.
        """

        rows: List[Dict[str, Any]] = []

        for ref_des, pin_name, net_name in self.edges:
            node = self.get_node(ref_des)

            if node is None:
                # Edge references unknown ref_des: drop row rather than invent data.
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
    Purpose:
        Convert user-typed net labels into a canonical `NET_*` token.

    Design:
        Idempotent for strings already prefixed with `NET_`; maps `NET-` to
        `NET_` for hyphenated imports.

    Workflow:
        Called when adding edges and when querying nets so UI and storage agree.

    Data handoff:
        Inputs: arbitrary user string; outputs single normalized net key stored
        on edges and echoed in ML CSV columns.
    """
    cleaned = raw_name.strip().upper().replace(" ", "_")

    if cleaned.startswith("NET_"):
        return cleaned

    if cleaned.startswith("NET-"):
        return cleaned.replace("NET-", "NET_")

    return f"NET_{cleaned}"
