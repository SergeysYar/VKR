from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, sqrt
from typing import Iterable

Vector3 = tuple[float, float, float]
Face = tuple[int, int, int]
Matrix4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _to_vector3(value: Iterable[float], field_name: str) -> Vector3:
    components = tuple(float(component) for component in value)
    if len(components) != 3:
        raise ValueError(f"{field_name} must contain exactly three components.")
    return components


def _to_face(value: Iterable[int]) -> Face:
    components = tuple(int(component) for component in value)
    if len(components) != 3:
        raise ValueError("face must contain exactly three vertex indices.")
    if any(index < 0 for index in components):
        raise ValueError("face indices must be >= 0.")
    return components


def _validate_face_indices(face: Face, vertex_count: int) -> None:
    if any(index >= vertex_count for index in face):
        raise ValueError("face index is out of range for current vertices.")


def _to_matrix4x4(value: Iterable[Iterable[float]]) -> Matrix4x4:
    rows = tuple(tuple(float(component) for component in row) for row in value)
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("matrix must be a 4x4 iterable of numbers.")
    return rows  # type: ignore[return-value]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v: Vector3) -> Vector3:
    length = sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if isclose(length, 0.0):
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _mul_mat3_vec3(m: tuple[tuple[float, float, float], ...], v: Vector3) -> Vector3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _inverse_transpose3x3(
    m: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]

    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(b * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - b * g)
    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d

    det = a * c00 + b * c01 + c * c02
    if isclose(det, 0.0):
        return m

    inv_t_scale = 1.0 / det
    return (
        (c00 * inv_t_scale, c10 * inv_t_scale, c20 * inv_t_scale),
        (c01 * inv_t_scale, c11 * inv_t_scale, c21 * inv_t_scale),
        (c02 * inv_t_scale, c12 * inv_t_scale, c22 * inv_t_scale),
    )


@dataclass
class Mesh:
    """
    Universal mesh container for export pipelines (for example OBJ).

    Faces store 0-based indices into vertices.
    """

    vertices: list[Vector3] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    normals: list[Vector3] | None = None

    def __post_init__(self) -> None:
        self.vertices = [_to_vector3(vertex, "vertex") for vertex in self.vertices]
        self.faces = [_to_face(face) for face in self.faces]
        for face in self.faces:
            _validate_face_indices(face, len(self.vertices))

        if self.normals is not None:
            self.normals = [_to_vector3(normal, "normal") for normal in self.normals]
            if len(self.normals) != len(self.vertices):
                raise ValueError("normals length must match vertices length.")

    def add_vertex(self, vertex: Iterable[float]) -> int:
        """Add a vertex and return its index."""
        self.vertices.append(_to_vector3(vertex, "vertex"))
        self.normals = None
        return len(self.vertices) - 1

    def add_face(self, face: Iterable[int]) -> int:
        """Add triangle face (i1, i2, i3) and return its index."""
        parsed_face = _to_face(face)
        _validate_face_indices(parsed_face, len(self.vertices))
        self.faces.append(parsed_face)
        self.normals = None
        return len(self.faces) - 1

    def merge(self, mesh: Mesh) -> None:
        """Merge another mesh into current mesh."""
        source = mesh
        if source is self:
            source = Mesh(
                vertices=list(self.vertices),
                faces=list(self.faces),
                normals=list(self.normals) if self.normals is not None else None,
            )

        index_offset = len(self.vertices)
        self.vertices.extend(source.vertices)
        self.faces.extend(
            (i1 + index_offset, i2 + index_offset, i3 + index_offset)
            for i1, i2, i3 in source.faces
        )

        if self.normals is not None and source.normals is not None:
            self.normals.extend(source.normals)
        else:
            self.normals = None

    def transform(self, matrix: Iterable[Iterable[float]]) -> None:
        """Apply 4x4 transform matrix to vertices and existing normals."""
        m = _to_matrix4x4(matrix)

        transformed_vertices: list[Vector3] = []
        for x, y, z in self.vertices:
            tx = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
            ty = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
            tz = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
            tw = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3]

            if not isclose(tw, 0.0):
                tx /= tw
                ty /= tw
                tz /= tw

            transformed_vertices.append((tx, ty, tz))

        self.vertices = transformed_vertices

        if self.normals is not None:
            normal_matrix = _inverse_transpose3x3(
                (
                    (m[0][0], m[0][1], m[0][2]),
                    (m[1][0], m[1][1], m[1][2]),
                    (m[2][0], m[2][1], m[2][2]),
                )
            )
            self.normals = [
                _normalize(_mul_mat3_vec3(normal_matrix, normal)) for normal in self.normals
            ]

    def compute_normals(self) -> list[Vector3]:
        """Compute and store per-vertex normals."""
        return compute_normals(self)


def compute_normals(mesh: Mesh) -> list[Vector3]:
    """Compute per-vertex normals from triangle faces and write to mesh.normals."""
    accum = [(0.0, 0.0, 0.0) for _ in mesh.vertices]

    for i1, i2, i3 in mesh.faces:
        v1 = mesh.vertices[i1]
        v2 = mesh.vertices[i2]
        v3 = mesh.vertices[i3]

        edge1 = _sub(v2, v1)
        edge2 = _sub(v3, v1)
        face_normal = _cross(edge1, edge2)

        for idx in (i1, i2, i3):
            x, y, z = accum[idx]
            accum[idx] = (
                x + face_normal[0],
                y + face_normal[1],
                z + face_normal[2],
            )

    normals = [_normalize(vector) for vector in accum]
    mesh.normals = normals
    return normals
