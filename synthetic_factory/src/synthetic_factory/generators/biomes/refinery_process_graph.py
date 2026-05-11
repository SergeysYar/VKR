from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Mapping


_ALLOWED_NODE_TYPES = {"column", "reactor", "tank", "exchanger", "pump", "valve"}
_ALLOWED_FLOW_DIRECTIONS = {"source_to_target", "target_to_source", "bidirectional"}


def _to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping.")
    return {str(key): inner for key, inner in value.items()}


def _parse_int_range(
    value: object,
    default_min: int,
    default_max: int,
    *,
    label: str,
) -> tuple[int, int]:
    if value is None:
        return (default_min, default_max)

    if isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise ValueError(f"{label} mapping must contain min/max.")
        minimum = int(value["min"])
        maximum = int(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        minimum = int(value[0])
        maximum = int(value[1])
    else:
        minimum = int(value)
        maximum = int(value)

    if minimum < 0 or maximum < 0:
        raise ValueError(f"{label} must be >= 0.")
    if minimum > maximum:
        raise ValueError(f"{label} min cannot be greater than max.")
    return (minimum, maximum)


def _sample_int(value_range: tuple[int, int], rng: Random) -> int:
    low, high = value_range
    if low == high:
        return low
    return rng.randint(low, high)


def _id_counter(ids: list[str], prefix: str) -> str:
    next_index = 1
    while True:
        candidate = f"{prefix}_{next_index}"
        if candidate not in ids:
            return candidate
        next_index += 1


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    parameters: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = self.type.strip().lower()
        if normalized not in _ALLOWED_NODE_TYPES:
            raise ValueError(f"Unsupported node type: {self.type}")
        node_id = self.id.strip()
        if not node_id:
            raise ValueError("Node id cannot be empty.")
        object.__setattr__(self, "type", normalized)
        object.__setattr__(self, "id", node_id)


@dataclass(frozen=True)
class Edge:
    source_id: str
    target_id: str
    pipe_parameters: dict[str, float | int] = field(default_factory=dict)
    flow_direction: str = "source_to_target"

    def __post_init__(self) -> None:
        source = self.source_id.strip()
        target = self.target_id.strip()
        direction = self.flow_direction.strip().lower()
        if not source or not target:
            raise ValueError("Edge source_id/target_id cannot be empty.")
        if direction not in _ALLOWED_FLOW_DIRECTIONS:
            raise ValueError(f"Unsupported flow_direction: {self.flow_direction}")
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "flow_direction", direction)


@dataclass
class ProcessGraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        if any(current.id == node.id for current in self.nodes):
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def node_ids(self) -> set[str]:
        return {node.id for node in self.nodes}

    def outgoing(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source_id == node_id]

    def incoming(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.target_id == node_id]

    @classmethod
    def generate_process_graph(
        cls,
        settings: Mapping[str, object] | None = None,
        *,
        seed: int = 0,
    ) -> ProcessGraph:
        return generate_process_graph(settings=settings, seed=seed)

    def validate_connectivity(self) -> tuple[bool, list[str]]:
        return validate_connectivity(self)


def generate_process_graph(
    settings: Mapping[str, object] | None = None,
    *,
    seed: int = 0,
) -> ProcessGraph:
    """
    Build a logical refinery process graph before geometry generation.

    Algorithm:
    1) main nodes: 1..3 distillation columns
    2) add reactors, exchangers, tanks
    3) connect columns -> reactors -> exchangers -> tanks
    4) add feedback loops
    5) add auxiliary nodes (pump/valve) on pipe routes
    """
    cfg = _to_mapping(settings)
    rng = Random(int(seed))

    nodes_cfg = _to_mapping(cfg.get("nodes"))
    columns_cfg = _to_mapping(nodes_cfg.get("columns"))
    reactors_cfg = _to_mapping(nodes_cfg.get("reactors"))
    exchangers_cfg = _to_mapping(nodes_cfg.get("exchangers"))
    tanks_cfg = _to_mapping(nodes_cfg.get("tanks"))
    aux_cfg = _to_mapping(nodes_cfg.get("auxiliary"))
    edges_cfg = _to_mapping(cfg.get("edges"))

    column_count = max(
        1,
        _sample_int(
            _parse_int_range(columns_cfg.get("count", [1, 3]), 1, 3, label="nodes.columns.count"),
            rng,
        ),
    )
    reactor_count = max(
        1,
        _sample_int(
            _parse_int_range(reactors_cfg.get("count", [1, 3]), 1, 3, label="nodes.reactors.count"),
            rng,
        ),
    )
    exchanger_count = max(
        1,
        _sample_int(
            _parse_int_range(exchangers_cfg.get("count", [1, 4]), 1, 4, label="nodes.exchangers.count"),
            rng,
        ),
    )
    tank_count = max(
        1,
        _sample_int(
            _parse_int_range(tanks_cfg.get("count", [1, 4]), 1, 4, label="nodes.tanks.count"),
            rng,
        ),
    )

    pump_per_link = max(
        0,
        _sample_int(
            _parse_int_range(aux_cfg.get("pump_per_link", [1, 1]), 1, 1, label="nodes.auxiliary.pump_per_link"),
            rng,
        ),
    )
    valve_per_link = max(
        0,
        _sample_int(
            _parse_int_range(aux_cfg.get("valve_per_link", [1, 1]), 1, 1, label="nodes.auxiliary.valve_per_link"),
            rng,
        ),
    )

    include_feedback_loops = bool(edges_cfg.get("include_feedback_loops", True))
    feedback_count = 0
    if include_feedback_loops:
        feedback_count = _sample_int(
            _parse_int_range(edges_cfg.get("feedback_loops", [1, 2]), 1, 2, label="edges.feedback_loops"),
            rng,
        )

    graph = ProcessGraph()

    for idx in range(column_count):
        graph.add_node(
            Node(
                id=f"column_{idx + 1}",
                type="column",
                parameters={"stage": idx + 1, "kind": "distillation"},
            )
        )
    for idx in range(reactor_count):
        graph.add_node(
            Node(
                id=f"reactor_{idx + 1}",
                type="reactor",
                parameters={"stage": idx + 1},
            )
        )
    for idx in range(exchanger_count):
        graph.add_node(
            Node(
                id=f"exchanger_{idx + 1}",
                type="exchanger",
                parameters={"stage": idx + 1},
            )
        )
    for idx in range(tank_count):
        graph.add_node(
            Node(
                id=f"tank_{idx + 1}",
                type="tank",
                parameters={"role": "product", "stage": idx + 1},
            )
        )

    known_node_ids = [node.id for node in graph.nodes]

    def add_aux_node(node_type: str, params: dict[str, float | int | str]) -> str:
        node_id = _id_counter(known_node_ids, node_type)
        graph.add_node(Node(id=node_id, type=node_type, parameters=params))
        known_node_ids.append(node_id)
        return node_id

    def add_pipe(source: str, target: str, *, radius: float, length: float, duty: str) -> None:
        graph.add_edge(
            Edge(
                source_id=source,
                target_id=target,
                pipe_parameters={
                    "radius": radius,
                    "length": length,
                    "duty": duty,
                },
                flow_direction="source_to_target",
            )
        )

    def connect_stage(
        source_id: str,
        target_id: str,
        *,
        radius: float,
        length: float,
        duty: str,
    ) -> None:
        current_source = source_id

        for idx in range(valve_per_link):
            valve_id = add_aux_node(
                "valve",
                params={
                    "size": round(max(0.2, radius * 2.4), 4),
                    "stage_index": idx + 1,
                    "duty": duty,
                },
            )
            add_pipe(current_source, valve_id, radius=radius, length=max(length * 0.18, 0.6), duty=duty)
            current_source = valve_id

        for idx in range(pump_per_link):
            pump_id = add_aux_node(
                "pump",
                params={
                    "size": round(max(0.35, radius * 2.8), 4),
                    "stage_index": idx + 1,
                    "duty": duty,
                },
            )
            add_pipe(current_source, pump_id, radius=radius, length=max(length * 0.2, 0.7), duty=duty)
            current_source = pump_id

        add_pipe(current_source, target_id, radius=radius, length=max(length * 0.62, 1.2), duty=duty)

    column_ids = [f"column_{idx + 1}" for idx in range(column_count)]
    reactor_ids = [f"reactor_{idx + 1}" for idx in range(reactor_count)]
    exchanger_ids = [f"exchanger_{idx + 1}" for idx in range(exchanger_count)]
    tank_ids = [f"tank_{idx + 1}" for idx in range(tank_count)]

    for idx, target_id in enumerate(reactor_ids):
        source_id = column_ids[idx % len(column_ids)]
        connect_stage(
            source_id,
            target_id,
            radius=0.24 + 0.01 * (idx % 3),
            length=3.6 + 0.35 * idx,
            duty="column_to_reactor",
        )

    for idx, target_id in enumerate(exchanger_ids):
        source_id = reactor_ids[idx % len(reactor_ids)]
        connect_stage(
            source_id,
            target_id,
            radius=0.2 + 0.01 * (idx % 2),
            length=3.0 + 0.3 * idx,
            duty="reactor_to_exchanger",
        )

    for idx, target_id in enumerate(tank_ids):
        source_id = exchanger_ids[idx % len(exchanger_ids)]
        connect_stage(
            source_id,
            target_id,
            radius=0.18 + 0.008 * (idx % 3),
            length=2.7 + 0.25 * idx,
            duty="exchanger_to_tank",
        )

    # Cross-branch manifolds keep parallel trains logically connected.
    def add_manifold_links(ids: list[str], *, radius: float, duty: str) -> None:
        if len(ids) <= 1:
            return
        for idx in range(len(ids) - 1):
            add_pipe(
                ids[idx],
                ids[idx + 1],
                radius=radius,
                length=1.9 + idx * 0.2,
                duty=duty,
            )

    add_manifold_links(column_ids, radius=0.17, duty="column_manifold")
    add_manifold_links(reactor_ids, radius=0.16, duty="reactor_manifold")
    add_manifold_links(exchanger_ids, radius=0.15, duty="exchanger_manifold")
    add_manifold_links(tank_ids, radius=0.14, duty="tank_manifold")

    if include_feedback_loops and feedback_count > 0:
        downstream_sources = [*tank_ids, *exchanger_ids]
        upstream_targets = [*column_ids, *reactor_ids]
        for idx in range(feedback_count):
            source_id = downstream_sources[idx % len(downstream_sources)]
            target_id = upstream_targets[idx % len(upstream_targets)]
            connect_stage(
                source_id,
                target_id,
                radius=0.16,
                length=4.2 + 0.5 * idx,
                duty="feedback_loop",
            )

    return graph


def validate_connectivity(graph: ProcessGraph) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not graph.nodes:
        return (False, ["Graph has no nodes."])

    node_ids = graph.node_ids()
    for edge in graph.edges:
        if edge.source_id not in node_ids:
            issues.append(f"Edge source is missing: {edge.source_id}")
        if edge.target_id not in node_ids:
            issues.append(f"Edge target is missing: {edge.target_id}")

    if issues:
        return (False, issues)

    if len(graph.nodes) > 1 and not graph.edges:
        return (False, ["Graph has multiple nodes but no edges."])

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    directed_out: dict[str, set[str]] = {node_id: set() for node_id in node_ids}

    for edge in graph.edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
        directed_out[edge.source_id].add(edge.target_id)
        if edge.flow_direction == "target_to_source":
            directed_out[edge.target_id].add(edge.source_id)
        elif edge.flow_direction == "bidirectional":
            directed_out[edge.target_id].add(edge.source_id)

    start = next(iter(node_ids))
    visited: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)

    disconnected = sorted(node_ids - visited)
    if disconnected:
        issues.append(f"Disconnected nodes: {', '.join(disconnected)}")

    for node in graph.nodes:
        incoming_count = len(graph.incoming(node.id))
        outgoing_count = len(graph.outgoing(node.id))
        if node.type in {"column", "reactor", "exchanger", "pump", "valve"}:
            if incoming_count == 0:
                issues.append(f"{node.type} {node.id} has no incoming flow.")
            if outgoing_count == 0:
                issues.append(f"{node.type} {node.id} has no outgoing flow.")
        elif node.type == "tank":
            if incoming_count == 0 and outgoing_count == 0:
                issues.append(f"tank {node.id} is isolated.")

    column_ids = [node.id for node in graph.nodes if node.type == "column"]
    reactor_ids = [node.id for node in graph.nodes if node.type == "reactor"]
    exchanger_ids = [node.id for node in graph.nodes if node.type == "exchanger"]
    tank_ids = [node.id for node in graph.nodes if node.type == "tank"]

    if column_ids and reactor_ids:
        if not any(_path_exists(directed_out, source_id, reactor_ids) for source_id in column_ids):
            issues.append("No directed flow path from any column to any reactor.")
    if reactor_ids and exchanger_ids:
        if not any(_path_exists(directed_out, source_id, exchanger_ids) for source_id in reactor_ids):
            issues.append("No directed flow path from any reactor to any exchanger.")
    if exchanger_ids and tank_ids:
        if not any(_path_exists(directed_out, source_id, tank_ids) for source_id in exchanger_ids):
            issues.append("No directed flow path from any exchanger to any tank.")

    return (len(issues) == 0, issues)


def _path_exists(adjacency: Mapping[str, set[str]], source_id: str, target_ids: list[str]) -> bool:
    targets = set(target_ids)
    visited: set[str] = set()
    stack = [source_id]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in targets and current != source_id:
            return True
        for neighbor in adjacency.get(current, set()):
            if neighbor not in visited:
                stack.append(neighbor)
    return False
