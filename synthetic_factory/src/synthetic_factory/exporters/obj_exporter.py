from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path
from typing import Generator

from ..geometry.mesh import Mesh, compute_normals
from ..scene.scene_graph import Scene, SceneObject, Transform

Matrix4 = list[list[float]]


def _identity_matrix() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat_mul(a: Matrix4, b: Matrix4) -> Matrix4:
    result: Matrix4 = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            result[row][col] = (
                a[row][0] * b[0][col]
                + a[row][1] * b[1][col]
                + a[row][2] * b[2][col]
                + a[row][3] * b[3][col]
            )
    return result


def _translation_matrix(x: float, y: float, z: float) -> Matrix4:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _scale_matrix(x: float, y: float, z: float) -> Matrix4:
    return [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotation_matrix_euler_deg(x: float, y: float, z: float) -> Matrix4:
    """
    Build rotation matrix from Euler degrees.

    Rotation order: Z * Y * X.
    """
    rx = radians(x)
    ry = radians(y)
    rz = radians(z)

    cx = cos(rx)
    sx = sin(rx)
    cy = cos(ry)
    sy = sin(ry)
    cz = cos(rz)
    sz = sin(rz)

    rot_x = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_y = [
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_z = [
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return _mat_mul(rot_z, _mat_mul(rot_y, rot_x))


def _local_matrix(transform: Transform) -> Matrix4:
    tx, ty, tz = transform.position
    rx, ry, rz = transform.rotation
    sx, sy, sz = transform.scale

    translation = _translation_matrix(tx, ty, tz)
    rotation = _rotation_matrix_euler_deg(rx, ry, rz)
    scale = _scale_matrix(sx, sy, sz)
    return _mat_mul(translation, _mat_mul(rotation, scale))


def _sanitize_name(name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"_", "-", ".", "/"} else "_"
        for ch in str(name)
    )
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"


def _clone_mesh(mesh: Mesh) -> Mesh:
    return Mesh(
        vertices=list(mesh.vertices),
        faces=list(mesh.faces),
        normals=list(mesh.normals) if mesh.normals is not None else None,
    )


class ObjExporter:
    """Export Scene graph to single OBJ file (v, vn, f, o, g)."""

    def export(self, scene: Scene, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = ["# synthetic_factory OBJ export"]
        vertex_offset = 0
        normal_offset = 0

        for obj, world_matrix, group_name in self._iter_objects(scene):
            if obj.mesh is None:
                continue

            mesh = _clone_mesh(obj.mesh)
            mesh.transform(world_matrix)
            if mesh.normals is None or len(mesh.normals) != len(mesh.vertices):
                compute_normals(mesh)

            lines.append(f"o {_sanitize_name(obj.id)}")
            lines.append(f"g {_sanitize_name(group_name)}")

            for x, y, z in mesh.vertices:
                lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")

            if mesh.normals is None:
                mesh.normals = [(0.0, 0.0, 0.0) for _ in mesh.vertices]
            for nx, ny, nz in mesh.normals:
                lines.append(f"vn {nx:.8f} {ny:.8f} {nz:.8f}")

            for i1, i2, i3 in mesh.faces:
                v1 = vertex_offset + i1 + 1
                v2 = vertex_offset + i2 + 1
                v3 = vertex_offset + i3 + 1

                n1 = normal_offset + i1 + 1
                n2 = normal_offset + i2 + 1
                n3 = normal_offset + i3 + 1
                lines.append(f"f {v1}//{n1} {v2}//{n2} {v3}//{n3}")

            vertex_offset += len(mesh.vertices)
            normal_offset += len(mesh.normals)

        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _iter_objects(
        self,
        scene: Scene,
    ) -> Generator[tuple[SceneObject, Matrix4, str], None, None]:
        identity = _identity_matrix()
        for root in scene.objects:
            yield from self._iter_object_tree(root, identity, [])

    def _iter_object_tree(
        self,
        obj: SceneObject,
        parent_matrix: Matrix4,
        parent_chain: list[str],
    ) -> Generator[tuple[SceneObject, Matrix4, str], None, None]:
        local = _local_matrix(obj.transform)
        world = _mat_mul(parent_matrix, local)
        chain = [*parent_chain, obj.id]
        group_name = "/".join(chain)

        yield obj, world, group_name

        for child in obj.children:
            yield from self._iter_object_tree(child, world, chain)
