from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .geometry import Mesh
from .scene import Scene, SceneObject, Transform


@dataclass
class _ObjMeshRecord:
    name: str
    group: str
    faces: list[tuple[int, int, int]] = field(default_factory=list)


def load_scene_from_obj(path: str) -> Scene:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"OBJ file not found: {source}")

    vertices: list[tuple[float, float, float]] = []
    records: list[_ObjMeshRecord] = []

    current_name = "unnamed_1"
    current_group = "default"
    current_record = _ObjMeshRecord(name=current_name, group=current_group)
    records.append(current_record)
    unnamed_counter = 1

    for raw_line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        keyword = parts[0].lower()
        if keyword == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue
        if keyword == "o":
            current_name = _clean_token(" ".join(parts[1:]) if len(parts) > 1 else f"unnamed_{unnamed_counter + 1}")
            unnamed_counter += 1
            current_record = _ObjMeshRecord(name=current_name, group=current_group)
            records.append(current_record)
            continue
        if keyword == "g":
            current_group = _clean_token(" ".join(parts[1:]) if len(parts) > 1 else "default")
            current_record.group = current_group
            continue
        if keyword == "f" and len(parts) >= 4:
            polygon = [_parse_face_index(token, len(vertices)) for token in parts[1:]]
            for triangle in _triangulate_polygon(polygon):
                current_record.faces.append(triangle)
            continue

    scene = Scene()
    object_index = 0
    for record in records:
        if not record.faces:
            continue
        mesh = _extract_mesh(vertices, record.faces)
        if not mesh.faces:
            continue
        object_index += 1
        object_name = record.name or f"obj_{object_index}"
        inferred_type = infer_object_type(object_name, record.group)
        scene.add_object(
            SceneObject(
                id=f"{object_name}_{object_index}",
                type=inferred_type,
                transform=Transform(),
                mesh=mesh,
            )
        )
    if not scene.objects:
        raise ValueError("OBJ does not contain any mesh faces.")
    return scene


def infer_object_type(object_name: str, group_name: str) -> str:
    token_source = f"{object_name} {group_name}".lower()
    normalized = token_source.replace("/", "_").replace("-", "_")
    tokens = [token for token in normalized.split("_") if token]

    if any(token in {"floor", "ground"} for token in tokens):
        return "floor"
    if any(token in {"ceiling", "roof"} for token in tokens):
        return "ceiling"
    if any(token in {"wall", "shell"} for token in tokens):
        return "wall"
    if any(token in {"pipe", "chimney", "duct"} for token in tokens):
        return "pipe"
    if any(token in {"wire", "cable"} for token in tokens):
        return "wire"
    if any(token in {"tray", "infrastructure", "junction", "support"} for token in tokens):
        return "infrastructure"
    if any(token in {"boiler", "tank"} for token in tokens):
        return "boiler"
    if any(token in {"desk", "chair", "monitor", "panel", "console"} for token in tokens):
        return "desk"
    if any(token in {"rack", "cabinet", "ups"} for token in tokens):
        return "rack"
    if any(token in {"conveyor", "belt"} for token in tokens):
        return "conveyor"
    if any(token in {"column", "beam", "frame", "structure"} for token in tokens):
        return "structure"
    if any(token in {"machine", "pump", "reactor", "valve", "equipment"} for token in tokens):
        return "machine"
    return "machine"


def _extract_mesh(
    global_vertices: list[tuple[float, float, float]],
    faces: Iterable[tuple[int, int, int]],
) -> Mesh:
    used_indices: set[int] = set()
    face_list = list(faces)
    for i1, i2, i3 in face_list:
        used_indices.add(i1)
        used_indices.add(i2)
        used_indices.add(i3)

    sorted_indices = sorted(used_indices)
    index_map = {old: new for new, old in enumerate(sorted_indices)}

    vertices = [global_vertices[index] for index in sorted_indices]
    remapped_faces = [
        (index_map[i1], index_map[i2], index_map[i3])
        for i1, i2, i3 in face_list
        if i1 in index_map and i2 in index_map and i3 in index_map
    ]
    return Mesh(vertices=vertices, faces=remapped_faces)


def _parse_face_index(token: str, vertex_count: int) -> int:
    head = token.split("/", 1)[0]
    index = int(head)
    if index == 0:
        raise ValueError("OBJ face indices cannot be 0.")
    if index < 0:
        resolved = vertex_count + index
    else:
        resolved = index - 1
    if resolved < 0 or resolved >= vertex_count:
        raise ValueError(f"OBJ face index out of range: {index}")
    return resolved


def _triangulate_polygon(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]
    triangles: list[tuple[int, int, int]] = []
    pivot = indices[0]
    for idx in range(1, len(indices) - 1):
        triangles.append((pivot, indices[idx], indices[idx + 1]))
    return triangles


def _clean_token(value: str) -> str:
    cleaned = "".join(char if (char.isalnum() or char in {"_", "-", "/"}) else "_" for char in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"
