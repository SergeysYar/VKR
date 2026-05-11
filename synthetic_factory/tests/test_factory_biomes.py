from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator


def _build_params() -> dict[str, object]:
    return {
        "factory_width": 120.0,
        "factory_depth": 90.0,
        "number_of_rooms": 8,
        "room_size_range": (8.0, 18.0),
        "corridor_width": 4.0,
        "layout_strategy": "perlin",
        "noise": {
            "seed": 42,
            "octaves": 4,
            "frequency": 1.0,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "sampling_scale": 0.06,
            "slot_multiplier": 2.0,
            "threshold": 0.3,
            "min_distance": 8.0,
            "room_margin_ratio": 0.9,
        },
        "biomes": {
            "enabled": True,
            "workshop_biome": "workshop",
            "cycle_order": [
                "workshop",
                "office",
                "boiler",
                "storage",
                "electrical",
                "maintenance",
                "laboratory",
                "control",
            ],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "conveyors_per_room": 3,
            "machines_per_room": 4,
        },
        "seed": 42,
    }


def test_layout_assigns_biomes_and_keeps_workshop_largest() -> None:
    generator = FactoryGenerator(_build_params())
    layout = generator.generate_layout()

    assert len(layout) == 8

    biomes = {room.biome for room in layout}
    assert biomes == {
        "workshop",
        "office",
        "boiler",
        "storage",
        "electrical",
        "maintenance",
        "laboratory",
        "control",
    }

    workshop_rooms = [room for room in layout if room.biome == "workshop"]
    assert len(workshop_rooms) == 1
    workshop_area = workshop_rooms[0].width * workshop_rooms[0].depth
    non_workshop_areas = [
        room.width * room.depth for room in layout if room.biome != "workshop"
    ]
    assert all(workshop_area + 1e-6 >= area for area in non_workshop_areas)


def test_scene_contains_biome_specific_content() -> None:
    generator = FactoryGenerator(_build_params())
    scene = generator.generate_scene()

    object_types = {obj.type for obj in scene.traverse()}
    expected_types = {
        "conveyor",
        "crane_bridge",
        "desk",
        "boiler_unit",
        "rack",
        "electrical_cabinet",
        "workbench",
        "lab_bench",
        "console",
    }

    assert expected_types.issubset(object_types)
    assert len(scene.find_by_type("room_workshop")) == 1


def test_biomes_have_different_room_sizes() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "grid",
            "number_of_rooms": 8,
            "room_size_range": (10.0, 10.0),
            "noise": {},
        }
    )

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()

    areas_by_biome = {
        spec.biome: round(spec.width * spec.depth, 3)
        for spec in layout
    }

    assert len(set(areas_by_biome.values())) > 1
    assert areas_by_biome["workshop"] > areas_by_biome["office"]
    assert areas_by_biome["boiler"] > areas_by_biome["control"]


def test_office_height_is_capped_by_profile() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "grid",
            "room_size_range": (20.0, 20.0),
            "room_height": 12.0,
            "noise": {},
        }
    )
    params["biomes"] = {
        **params["biomes"],  # type: ignore[index]
        "ensure_all_types": True,
    }

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    office = next(room for room in layout if room.biome == "office")

    assert generator._effective_room_height(office) <= 4.2 + 1e-6


def test_workshop_area_ratio_creates_connected_cluster_when_configured() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "grid",
            "room_size_range": (10.0, 10.0),
            "noise": {},
        }
    )
    params["biomes"] = {
        **params["biomes"],  # type: ignore[index]
        "workshop_area_ratio": 0.75,
        "clustered_assignment": True,
    }

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    workshop_rooms = [spec for spec in layout if spec.biome == "workshop"]

    total_area = sum(spec.width * spec.depth for spec in layout)
    workshop_area = sum(spec.width * spec.depth for spec in workshop_rooms)
    assert workshop_area / total_area >= 0.75 - 1e-6

    workshop_cells = {(spec.row, spec.col) for spec in workshop_rooms}
    start = next(iter(workshop_cells))
    visited: set[tuple[int, int]] = set()
    stack = [start]
    while stack:
        row, col = stack.pop()
        if (row, col) in visited:
            continue
        visited.add((row, col))
        for cell in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if cell in workshop_cells and cell not in visited:
                stack.append(cell)

    assert visited == workshop_cells


def test_ensure_all_types_keeps_all_biomes_even_with_high_workshop_ratio() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "grid",
            "number_of_rooms": 8,
            "room_size_range": (10.0, 10.0),
            "noise": {},
        }
    )
    params["biomes"] = {
        **params["biomes"],  # type: ignore[index]
        "ensure_all_types": True,
        "workshop_area_ratio": 0.75,
    }

    layout = FactoryGenerator(params).generate_layout()
    biomes = {spec.biome for spec in layout}

    assert biomes == {
        "workshop",
        "office",
        "boiler",
        "storage",
        "electrical",
        "maintenance",
        "laboratory",
        "control",
    }


def test_room_doors_are_placed_on_sides_with_neighbor_rooms() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "grid",
            "number_of_rooms": 4,
            "room_size_range": (10.0, 10.0),
            "noise": {},
        }
    )
    params["biomes"] = {
        **params["biomes"],  # type: ignore[index]
        "inter_room_doors": True,
    }

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    scene = generator.instantiate_rooms()

    for spec in layout:
        expected_sides = set(generator._door_sides_for_room(spec))  # noqa: SLF001
        room_root = scene.get_object(spec.room_id)
        assert room_root is not None

        door_objects = [child for child in room_root.children if child.type == "door_opening"]
        door_ids = [child.id for child in door_objects]
        for side in expected_sides:
            assert any(f"_door_opening_{side}_" in door_id for door_id in door_ids)
        assert all(door.mesh is None for door in door_objects)


def test_layout_has_no_empty_slots_in_bounding_grid() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "perlin",
            "number_of_rooms": 8,
            "room_size_range": (10.0, 10.0),
            "noise": {
                **params["noise"],  # type: ignore[index]
                "full_occupancy": True,
                "connectivity_bias": 1.0,
            },
        }
    )

    layout = FactoryGenerator(params).generate_layout()
    cells = {(spec.row, spec.col) for spec in layout}
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    assert len(cells) == len(layout)

    expected_cells = {
        (row, col)
        for row in range(min(rows), max(rows) + 1)
        for col in range(min(cols), max(cols) + 1)
    }
    assert cells == expected_cells


def test_industrial_rooms_are_significantly_larger_than_non_industrial() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "grid",
            "room_size_range": (10.0, 10.0),
            "noise": {"full_occupancy": True},
        }
    )
    params["biomes"] = {
        **params["biomes"],  # type: ignore[index]
        "industrial_size_bias": True,
        "industrial_size_multiplier": 1.4,
        "non_industrial_size_multiplier": 0.5,
        "industrial_min_size_ratio": 0.8,
        "industrial_max_size_ratio": 0.99,
        "non_industrial_min_size_ratio": 0.25,
        "non_industrial_max_size_ratio": 0.55,
    }

    layout = FactoryGenerator(params).generate_layout()
    areas = {spec.biome: spec.width * spec.depth for spec in layout}
    industrial = [areas[name] for name in ("workshop", "boiler", "storage", "electrical", "maintenance", "laboratory")]
    non_industrial = [areas[name] for name in ("office", "control")]

    assert min(industrial) >= max(non_industrial) * 2.0


def test_door_openings_keep_lintel_above_passage() -> None:
    params = _build_params()
    params.update(
        {
            "layout_strategy": "grid",
            "number_of_rooms": 4,
            "room_size_range": (10.0, 10.0),
            "noise": {},
        }
    )
    params["biomes"] = {
        **params["biomes"],  # type: ignore[index]
        "inter_room_doors": True,
        "door_height": 2.2,
    }

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    scene = generator.instantiate_rooms()

    for spec in layout:
        expected_sides = set(generator._door_sides_for_room(spec))  # noqa: SLF001
        if not expected_sides:
            continue
        room_root = scene.get_object(spec.room_id)
        assert room_root is not None
        room_height = generator._effective_room_height(spec)  # noqa: SLF001

        lintels = [child for child in room_root.children if "_lintel_" in child.id]
        assert len(lintels) >= len(expected_sides)
        assert all(lintel.transform.position[2] > room_height / 2.0 for lintel in lintels)
