from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator
from synthetic_factory.generators.biomes.boiler import (
    HeightConstraint,
    MinClearanceRule,
    PipeConnectivityRule,
    PlatformSupportRule,
    _PipeConnectivityState,
)


def _build_params() -> dict[str, object]:
    return {
        "factory_width": 24.0,
        "factory_depth": 20.0,
        "number_of_rooms": 1,
        "room_size_range": (14.0, 14.0),
        "corridor_width": 1.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {
            "enabled": True,
            "workshop_biome": "boiler",
            "cycle_order": ["boiler"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "boiler": {
                "pattern": "linear_boilers",
                "random_variation": False,
                "rules": {
                    "auto_fix": True,
                    "min_clearance": 1.2,
                },
                "min_walkway": 1.2,
                "service_clearance": 0.8,
                "edge_offset": 1.0,
                "boiler": {
                    "count": [3, 3],
                    "height": [8.0, 8.0],
                    "radius": [1.0, 1.0],
                },
                "pipes": {
                    "density": [0.6, 0.6],
                    "radius": [0.2, 0.2],
                },
                "platforms": {
                    "levels": [2, 2],
                    "width_factor": [1.5, 1.5],
                },
                "tanks": {
                    "count": [0, 0],
                },
                "structure": {
                    "beam_density": [0.6, 0.6],
                },
            },
        },
        "seed": 42,
    }


def test_min_clearance_rule_auto_spreads_positions() -> None:
    rule = MinClearanceRule(minimum_distance=1.0)
    positions, fixed = rule.enforce_axis_spacing(
        axis_positions=[-0.2, 0.0, 0.2],
        axis_limit=2.0,
        object_radius=0.5,
    )

    assert fixed
    required = 2.0 * 0.5 + 1.0
    for idx in range(len(positions) - 1):
        assert positions[idx + 1] - positions[idx] + 1e-9 >= required


def test_height_constraint_clamps_object_pose() -> None:
    rule = HeightConstraint(max_height=5.0, floor_height=0.0, margin=0.1)
    z, height, changed = rule.clamp_pose(center_z=4.9, object_height=2.0)
    assert changed
    assert height <= 4.8
    assert z + height / 2.0 <= 5.0 + 1e-9


def test_pipe_connectivity_rule_autofix_adds_missing_segments() -> None:
    rule = PipeConnectivityRule()
    state = _PipeConnectivityState(riser_axes=set(), has_main_manifold=False, link_count=0)
    emitted: list[str] = []

    def axis_to_xy(axis_value: float, lateral: float = 0.0) -> tuple[float, float]:
        return (axis_value, lateral)

    def emit(
        object_type: str,
        mesh: object,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        nominal_height: float | None,
    ) -> None:
        del mesh, position, rotation, nominal_height
        emitted.append(object_type)

    fixes = rule.ensure_connectivity(
        state=state,
        boiler_axes=[-2.0, 2.0],
        axis_to_xy=axis_to_xy,
        axis_rotation=(0.0, 0.0, 0.0),
        emit=emit,
        boiler_height=3.0,
        pipe_z=5.0,
        pipe_radius=0.2,
    )

    assert fixes >= 4
    assert emitted.count("riser_pipe") == 2
    assert "pipe_manifold_main" in emitted
    assert "pipe_link" in emitted


def test_platform_support_rule_adds_four_supports() -> None:
    rule = PlatformSupportRule(support_radius=0.08)
    height_rule = HeightConstraint(max_height=10.0)
    emitted_types: list[str] = []

    def emit(
        object_type: str,
        mesh: object,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        nominal_height: float | None,
    ) -> None:
        del mesh, position, rotation, nominal_height
        emitted_types.append(object_type)

    added = rule.ensure_supports(
        platform_center=(0.0, 0.0, 3.0),
        platform_width=4.0,
        platform_depth=3.0,
        platform_thickness=0.2,
        height_constraint=height_rule,
        emit=emit,
    )

    assert added == 4
    assert emitted_types == ["platform_support"] * 4


def test_boiler_scene_contains_access_and_platform_supports() -> None:
    generator = FactoryGenerator(_build_params())
    layout = generator.generate_layout()
    room_height = generator._effective_room_height(layout[0])  # noqa: SLF001

    scene = generator.generate_scene()
    assert len(scene.find_by_type("access_corridor")) >= 1

    platform_count = len(scene.find_by_type("service_platform"))
    support_count = len(scene.find_by_type("platform_support"))
    assert support_count >= platform_count * 4

    boiler_count = len(scene.find_by_type("boiler_unit"))
    riser_count = len(scene.find_by_type("riser_pipe"))
    assert riser_count >= boiler_count
    assert len(scene.find_by_type("pipe_manifold_main")) >= 1
    if boiler_count > 1:
        assert len(scene.find_by_type("pipe_link")) >= 1

    for obj in scene.traverse():
        z = obj.transform.position[2]
        assert z >= -1e-6
        assert z <= room_height + 1e-6

