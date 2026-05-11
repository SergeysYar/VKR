from __future__ import annotations

from synthetic_factory.generators.biomes.refinery_process_graph import (
    Edge,
    Node,
    ProcessGraph,
    generate_process_graph,
    validate_connectivity,
)


def _path_exists(edges: list[Edge], source_prefix: str, target_prefix: str) -> bool:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_id, set()).add(edge.target_id)

    starts = sorted(node_id for node_id in adjacency if node_id.startswith(source_prefix))
    targets = {node_id for node_id in adjacency.keys() if node_id.startswith(target_prefix)}
    for edge in edges:
        if edge.target_id.startswith(target_prefix):
            targets.add(edge.target_id)

    for start in starts:
        visited: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current != start and current in targets:
                return True
            stack.extend(next_id for next_id in adjacency.get(current, set()) if next_id not in visited)
    return False


def test_generate_process_graph_contains_required_node_types() -> None:
    graph = generate_process_graph(seed=7)
    node_types = {node.type for node in graph.nodes}
    assert {"column", "reactor", "tank", "exchanger", "pump", "valve"}.issubset(node_types)
    column_count = sum(1 for node in graph.nodes if node.type == "column")
    assert 1 <= column_count <= 3
    assert graph.edges
    assert all(edge.flow_direction == "source_to_target" for edge in graph.edges)
    assert _path_exists(graph.edges, "column_", "reactor_")
    assert _path_exists(graph.edges, "reactor_", "exchanger_")
    assert _path_exists(graph.edges, "exchanger_", "tank_")
    assert any(edge.pipe_parameters.get("duty") == "feedback_loop" for edge in graph.edges)


def test_generate_process_graph_is_deterministic_by_seed() -> None:
    left = generate_process_graph(seed=42)
    right = generate_process_graph(seed=42)

    left_nodes = sorted((node.id, node.type) for node in left.nodes)
    right_nodes = sorted((node.id, node.type) for node in right.nodes)
    left_edges = sorted((edge.source_id, edge.target_id) for edge in left.edges)
    right_edges = sorted((edge.source_id, edge.target_id) for edge in right.edges)

    assert left_nodes == right_nodes
    assert left_edges == right_edges


def test_validate_connectivity_passes_for_generated_graph() -> None:
    graph = generate_process_graph(seed=9)
    ok, issues = validate_connectivity(graph)
    assert ok
    assert not issues


def test_validate_connectivity_fails_for_missing_node_reference() -> None:
    graph = ProcessGraph(
        nodes=[Node(id="column_1", type="column"), Node(id="tank_1", type="tank")],
        edges=[Edge(source_id="column_1", target_id="ghost_node")],
    )
    ok, issues = graph.validate_connectivity()
    assert not ok
    assert any("missing" in issue for issue in issues)


def test_validate_connectivity_fails_for_disconnected_node() -> None:
    graph = ProcessGraph(
        nodes=[
            Node(id="tank_1", type="tank"),
            Node(id="reactor_1", type="reactor"),
            Node(id="column_1", type="column"),
        ],
        edges=[Edge(source_id="tank_1", target_id="reactor_1")],
    )
    ok, issues = graph.validate_connectivity()
    assert not ok
    assert any("Disconnected nodes" in issue for issue in issues)


def test_validate_connectivity_detects_stage_breaks() -> None:
    graph = ProcessGraph(
        nodes=[
            Node(id="column_1", type="column"),
            Node(id="valve_1", type="valve"),
            Node(id="pump_1", type="pump"),
            Node(id="reactor_1", type="reactor"),
            Node(id="exchanger_1", type="exchanger"),
            Node(id="tank_1", type="tank"),
        ],
        edges=[
            Edge(source_id="column_1", target_id="valve_1"),
            Edge(source_id="valve_1", target_id="pump_1"),
            Edge(source_id="pump_1", target_id="reactor_1"),
            # broken: no reactor -> exchanger
            Edge(source_id="exchanger_1", target_id="tank_1"),
        ],
    )
    ok, issues = graph.validate_connectivity()
    assert not ok
    assert any("reactor" in issue and "exchanger" in issue for issue in issues)
