from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator, Iterable, Literal

if TYPE_CHECKING:
    from ..geometry.mesh import Mesh

Vector3 = tuple[float, float, float]
AnchorType = Literal["top", "side", "bottom"]
_VALID_ANCHOR_TYPES = {"top", "side", "bottom"}


def _to_vector3(value: Iterable[float], field_name: str) -> Vector3:
    values = tuple(float(component) for component in value)
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly three components.")
    return values


@dataclass(frozen=True)
class AnchorPoint:
    """Anchor point in object-local space used for logical attachments."""

    position: Vector3
    type: AnchorType
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _to_vector3(self.position, "anchor position"))
        normalized_type = str(self.type).strip().lower()
        if normalized_type not in _VALID_ANCHOR_TYPES:
            raise ValueError("anchor type must be one of: top, side, bottom.")
        object.__setattr__(self, "type", normalized_type)
        object.__setattr__(self, "name", str(self.name).strip())


@dataclass
class Transform:
    """Local transform for an object in 3D space."""

    position: Vector3 = (0.0, 0.0, 0.0)
    rotation: Vector3 = (0.0, 0.0, 0.0)  # Euler angles, degrees/radians defined by exporter
    scale: Vector3 = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        self.position = _to_vector3(self.position, "position")
        self.rotation = _to_vector3(self.rotation, "rotation")
        self.scale = _to_vector3(self.scale, "scale")


@dataclass
class SceneObject:
    """Node in scene hierarchy, similar to Unity/Blender object tree."""

    id: str
    type: str
    transform: Transform = field(default_factory=Transform)
    mesh: Mesh | None = None
    anchor_points: list[AnchorPoint] = field(default_factory=list)
    children: list[SceneObject] = field(default_factory=list)
    parent: SceneObject | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.anchor_points = list(self.anchor_points)
        initial_children = list(self.children)
        self.children = []
        for child in initial_children:
            self.add_child(child)
        self.ensure_anchor_points()

    def add_child(self, child: SceneObject) -> None:
        """Attach child to this object with cycle protection."""
        if child is self:
            raise ValueError("Object cannot be parent of itself.")
        if self._is_descendant_of(child):
            raise ValueError("Cycle detected in scene hierarchy.")

        if child.parent is not None:
            child.parent.remove_child(child.id)

        child.parent = self
        self.children.append(child)

    def remove_child(self, object_id: str, recursive: bool = False) -> SceneObject | None:
        """Remove direct or nested child by id."""
        for index, child in enumerate(self.children):
            if child.id == object_id:
                removed = self.children.pop(index)
                removed.parent = None
                return removed

        if not recursive:
            return None

        for child in self.children:
            removed = child.remove_child(object_id, recursive=True)
            if removed is not None:
                return removed

        return None

    def traverse(self) -> Generator[SceneObject, None, None]:
        """Depth-first traversal from this node."""
        yield self
        for child in self.children:
            yield from child.traverse()

    def add_anchor_point(self, anchor: AnchorPoint) -> None:
        """Add anchor to the object."""
        self.anchor_points.append(anchor)

    def anchors_by_type(self, anchor_type: AnchorType) -> list[AnchorPoint]:
        """Return anchors of specified type."""
        normalized_type = str(anchor_type).strip().lower()
        if normalized_type not in _VALID_ANCHOR_TYPES:
            raise ValueError("anchor_type must be one of: top, side, bottom.")
        return [anchor for anchor in self.anchor_points if anchor.type == normalized_type]

    def ensure_anchor_points(self) -> None:
        """Generate default anchors from mesh bounds when none are provided."""
        if self.anchor_points or self.mesh is None or not self.mesh.vertices:
            return
        self.anchor_points = self._default_anchor_points_from_mesh()

    def _is_descendant_of(self, candidate_ancestor: SceneObject) -> bool:
        current = self.parent
        while current is not None:
            if current is candidate_ancestor:
                return True
            current = current.parent
        return False

    def _default_anchor_points_from_mesh(self) -> list[AnchorPoint]:
        assert self.mesh is not None
        xs = [vertex[0] for vertex in self.mesh.vertices]
        ys = [vertex[1] for vertex in self.mesh.vertices]
        zs = [vertex[2] for vertex in self.mesh.vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = (min_z + max_z) / 2.0

        return [
            AnchorPoint(position=(center_x, center_y, max_z), type="top", name="top_center"),
            AnchorPoint(position=(center_x, center_y, min_z), type="bottom", name="bottom_center"),
            AnchorPoint(position=(max_x, center_y, center_z), type="side", name="side_pos_x"),
            AnchorPoint(position=(min_x, center_y, center_z), type="side", name="side_neg_x"),
            AnchorPoint(position=(center_x, max_y, center_z), type="side", name="side_pos_y"),
            AnchorPoint(position=(center_x, min_y, center_z), type="side", name="side_neg_y"),
        ]


@dataclass
class Scene:
    """Whole scene storage before export to OBJ (or other formats)."""

    objects: list[SceneObject] = field(default_factory=list)

    def add_object(self, obj: SceneObject, parent_id: str | None = None) -> None:
        """
        Add object to scene root or under parent.

        parent_id is optional to keep a simple add_object(obj) API while supporting hierarchy.
        """
        existing = self.get_object(obj.id)
        if existing is not None and existing is not obj:
            raise ValueError(f"Object with id '{obj.id}' already exists in scene.")

        self._detach_if_attached(obj)

        if parent_id is None:
            obj.parent = None
            self.objects.append(obj)
            return

        parent = self.get_object(parent_id)
        if parent is None:
            raise ValueError(f"Parent with id '{parent_id}' not found.")
        parent.add_child(obj)

    def remove_object(self, object_id: str) -> SceneObject | None:
        """Remove object by id from root or nested hierarchy."""
        for index, root in enumerate(self.objects):
            if root.id == object_id:
                removed = self.objects.pop(index)
                removed.parent = None
                return removed

        for root in self.objects:
            removed = root.remove_child(object_id, recursive=True)
            if removed is not None:
                return removed

        return None

    def find_by_type(self, object_type: str) -> list[SceneObject]:
        """Find all objects by type (room, wall, machine, column, ...)."""
        return [obj for obj in self.traverse() if obj.type == object_type]

    def traverse(self) -> Generator[SceneObject, None, None]:
        """Depth-first traversal for all root objects."""
        for root in self.objects:
            yield from root.traverse()

    def group_objects(
        self,
        group_id: str,
        object_ids: Iterable[str],
        group_type: str = "group",
    ) -> SceneObject:
        """Group existing scene objects under a new group node."""
        if self.get_object(group_id) is not None:
            raise ValueError(f"Object with id '{group_id}' already exists in scene.")

        target_objects: list[SceneObject] = []
        for object_id in object_ids:
            obj = self.get_object(object_id)
            if obj is None:
                raise ValueError(f"Object with id '{object_id}' not found.")
            target_objects.append(obj)

        if not target_objects:
            raise ValueError("group_objects requires at least one object id.")

        group = SceneObject(id=group_id, type=group_type)

        common_parent = target_objects[0].parent
        if any(obj.parent is not common_parent for obj in target_objects):
            common_parent = None

        if common_parent is None:
            self.objects.append(group)
        else:
            common_parent.add_child(group)

        for obj in target_objects:
            self._detach_if_attached(obj)
            group.add_child(obj)

        return group

    def get_object(self, object_id: str) -> SceneObject | None:
        """Find object by id."""
        for obj in self.traverse():
            if obj.id == object_id:
                return obj
        return None

    def _detach_if_attached(self, obj: SceneObject) -> None:
        if obj.parent is not None:
            obj.parent.remove_child(obj.id)
            return

        for index, root in enumerate(self.objects):
            if root is obj:
                self.objects.pop(index)
                return
