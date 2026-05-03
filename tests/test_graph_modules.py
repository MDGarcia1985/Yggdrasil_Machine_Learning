"""
test_graph_modules.py

Purpose:
Smoke-test the circuit graph, node, edge, simulator, and graph styling modules.

Validates:
- default graph construction
- pin-to-net edge normalization
- graph-to-row export for ML/preprocess handoff
- lightweight simulator validation output
- node/edge/graph styling helpers
"""

from src.edges import init_default_edges
from src.graph import CircuitGraph, normalize_net_name
from src.nodes import Node, Pin, init_default_nodes
from src.simulator import (
    mark_node_complete,
    mark_prediction_accepted,
    mark_prediction_rejected,
    set_role,
    tick_sim,
)
from ui.streamlit_ui.edge_styling import get_edge_style
from ui.streamlit_ui.graph_styling import get_graph_config
from ui.streamlit_ui.node_styling import get_node_label, get_node_style
from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run graph/module sniff test for the circuit graph layer.
    """
    print("Graph Modules Sniff Test (Circuit Dataset)")

    graph = CircuitGraph(
        circuit_name="button_debounce",
        description="RC debounce with schmitt trigger",
        notes="Smoke-test graph.",
    )
    graph.load_default()

    rows = graph.to_rows()
    validation = tick_sim(graph)
    nodes = init_default_nodes()
    edges = init_default_edges()
    first_node = graph.get_node("R1")
    first_edge = graph.get_edges_list()[0]
    edge_style = get_edge_style(*first_edge, graph=graph)
    graph_config = get_graph_config("Technical")

    print("\nDefault Nodes:")
    print(list(nodes.keys()))

    print("\nDefault Edge Count:")
    print(len(edges))

    print("\nGraph Rows Sample:")
    for row in rows[:5]:
        print(row)

    print("\nValidation:")
    print(validation)

    print("\nStyling:")
    print(get_node_style(first_node, "Technical"))
    print(get_node_label(first_node, "Technical"))
    print(edge_style)
    print(graph_config)

    print("\nNormalization:")
    print(normalize_net_name("vin"))
    print(normalize_net_name("NET-GND"))

    set_role(graph, "R1", "FILTER")
    mark_node_complete(graph, "R1")
    mark_prediction_accepted(graph, "R1", "capacitor")
    accepted_state = graph.get_node("R1").state
    mark_prediction_rejected(graph, "R1", "capacitor")
    rejected_state = graph.get_node("R1").state

    manual = CircuitGraph("manual")
    manual.add_node(
        Node(
            ref_des="r2",
            component_type="resistor",
            component_kind="primitive",
            pins=[
                Pin(name="A", role="PASSIVE", net="NET_VIN"),
                Pin(name="B", role="PASSIVE", net="NET_GND"),
            ],
        )
    )
    manual.add_edge("r2", "a", "vin")
    manual.add_edge("r2", "b", "gnd")

    print("\nManual Graph:")
    print(manual.get_node("R2"))
    print(manual.get_edges_for_node("r2"))
    print(manual.to_rows())

    print("\nBasic Consistency Checks:")
    print(f"Default graph has nodes: {len(graph.nodes) > 0}")
    print(f"Default graph has edges: {len(graph.edges) > 0}")
    print(f"Graph rows created: {len(rows) == len(graph.edges)}")
    print(f"Validation has graph_state: {'graph_state' in validation}")
    print(f"Power net normalized: {normalize_net_name('vin') == 'NET_VIN'}")
    print(f"Accepted state recorded: {accepted_state == 'VALIDATED'}")
    print(f"Rejected state recorded: {rejected_state == 'REVIEW'}")
    print(f"Manual ref lookup normalized: {manual.get_node('R2') is not None}")


def test_graph_modules_smoke() -> None:
    output = capture_output(run_test)
    assert "Graph Modules Sniff Test" in output
    assert "Graph rows created: True" in output
    assert "Validation has graph_state: True" in output


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_graph_modules", output)
