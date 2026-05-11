from __future__ import annotations

from pathlib import Path

from synthetic_factory.parametric.parameters import ParameterSet
from synthetic_factory.pipeline import (
    AuxiliaryStage,
    InfrastructureStage,
    LayoutGenerationStage,
    ParameterSamplingStage,
    PipelineContext,
    RoomGenerationStage,
    SceneAssemblyStage,
    build_default_scene_pipeline,
)


def _pipeline_parameter_set(seed: int) -> ParameterSet:
    return ParameterSet(
        {
            "factory_width": 28.0,
            "factory_depth": 22.0,
            "number_of_rooms": 1,
            "corridor_width": 1.0,
            "room_height": 4.0,
            "room_size_min": 14.0,
            "room_size_max": 14.0,
        },
        seed=seed,
    )


def _pipeline_base_parameters(seed: int) -> dict[str, object]:
    return {
        "factory_width": 28.0,
        "factory_depth": 22.0,
        "number_of_rooms": 1,
        "corridor_width": 1.0,
        "room_height": 4.0,
        "layout_strategy": "grid",
        "room_size_min": 14.0,
        "room_size_max": 14.0,
        "room_count_range": [1, 1],
        "noise": {"seed": seed},
        "biomes": {
            "enabled": True,
            "workshop_biome": "laboratory",
            "cycle_order": ["laboratory"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "auxiliary": {
                "enabled": True,
                "density": 1.0,
                "random_variation": False,
            },
            "infrastructure": {
                "enabled": True,
                "include_pipes": True,
                "tray_height": 2.8,
                "tray_width": 0.3,
                "tray_thickness": 0.1,
                "room_link_width": 0.24,
            },
            "laboratory": {
                "seed": 13,
                "random_variation": False,
                "stations": {
                    "count": [4, 4],
                    "spacing": [1.8, 1.8],
                    "type_variability": [0.2, 0.2],
                },
                "equipment": {
                    "density": [2.0, 2.0],
                },
            },
        },
        "seed": seed,
    }


def _index_of_log(logs: list[str], prefix: str) -> int:
    return next(index for index, value in enumerate(logs) if value.startswith(prefix))


def test_default_pipeline_has_infrastructure_and_auxiliary_stages_in_order(tmp_path: Path) -> None:
    seed = 42
    export_path = tmp_path / "pipeline_scene.obj"
    pipeline = build_default_scene_pipeline(
        parameter_set=_pipeline_parameter_set(seed),
        export_path=str(export_path),
        seed=seed,
        base_parameters=_pipeline_base_parameters(seed),
    )

    result = pipeline.run()

    assembly_idx = _index_of_log(result.logs, "[SceneAssembly]")
    infrastructure_idx = _index_of_log(result.logs, "[Infrastructure]")
    auxiliary_idx = _index_of_log(result.logs, "[Auxiliary]")
    export_idx = _index_of_log(result.logs, "[Export]")

    assert assembly_idx < infrastructure_idx < auxiliary_idx < export_idx
    assert export_path.exists()
    assert result.scene is not None
    assert result.scene.get_object("global_infrastructure") is not None
    assert len([obj for obj in result.scene.traverse() if obj.type.startswith("aux_")]) > 0


def test_infrastructure_stage_preserves_existing_geometry() -> None:
    seed = 7
    context = PipelineContext(seed=seed)
    parameter_set = _pipeline_parameter_set(seed)
    base_parameters = _pipeline_base_parameters(seed)

    context = ParameterSamplingStage(parameter_set=parameter_set).run(context)
    context = LayoutGenerationStage(base_parameters=base_parameters).run(context)
    context = RoomGenerationStage().run(context)
    context = SceneAssemblyStage().run(context)
    assert context.scene is not None

    floor_ids_before = {obj.id for obj in context.scene.find_by_type("floor")}
    wall_ids_before = {obj.id for obj in context.scene.find_by_type("wall")}
    ceiling_ids_before = {obj.id for obj in context.scene.find_by_type("ceiling")}

    context = InfrastructureStage().run(context)
    assert context.scene is not None
    floor_ids_after = {obj.id for obj in context.scene.find_by_type("floor")}
    wall_ids_after = {obj.id for obj in context.scene.find_by_type("wall")}
    ceiling_ids_after = {obj.id for obj in context.scene.find_by_type("ceiling")}

    assert floor_ids_after == floor_ids_before
    assert wall_ids_after == wall_ids_before
    assert ceiling_ids_after == ceiling_ids_before
    assert context.scene.get_object("global_infrastructure") is not None

    context = AuxiliaryStage().run(context)
    assert len([obj for obj in context.scene.traverse() if obj.type.startswith("aux_")]) > 0
