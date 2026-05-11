from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Protocol

from ..exporters.obj_exporter import ObjExporter
from ..generators.factory_generator import FactoryGenerator, FactoryParams, RoomLayout
from ..parametric.parameters import ParameterSet
from ..scene.scene_graph import Scene
from ..sensing import LidarStation, LidarSurveyGenerator, LidarSurveyResult


def _to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _split_name_and_suffix(path: Path) -> tuple[str, str]:
    suffix = "".join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    return stem, suffix


def _with_suffix_token(path: Path, token: str) -> Path:
    stem, suffix = _split_name_and_suffix(path)
    return path.with_name(f"{stem}_{token}{suffix}")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = _split_name_and_suffix(path)
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


class PipelineStage(Protocol):
    def run(self, context: PipelineContext) -> PipelineContext:
        ...


@dataclass
class PipelineContext:
    seed: int | None = 0
    sampled_parameters: dict[str, float] = field(default_factory=dict)
    resolved_factory_parameters: dict[str, object] = field(default_factory=dict)
    factory_params: FactoryParams | None = None
    layout: list[RoomLayout] = field(default_factory=list)
    scene: Scene | None = None
    factory_generator: FactoryGenerator | None = None
    export_path: str | None = None
    lidar_station_records: list[dict[str, object]] = field(default_factory=list)
    lidar_circle_records: list[dict[str, object]] = field(default_factory=list)
    lidar_labels: dict[str, int] = field(default_factory=dict)
    lidar_output_dir: str | None = None
    lidar_circle_file_paths: list[str] = field(default_factory=list)
    lidar_stitched_cloud_path: str | None = None
    lidar_stitched_cloud_interior_path: str | None = None
    lidar_stitched_cloud_exterior_path: str | None = None
    lidar_room_stitched_paths: dict[str, str] = field(default_factory=dict)
    lidar_biome_stitched_paths: dict[str, str] = field(default_factory=dict)
    lidar_dataset_path: str | None = None
    logs: list[str] = field(default_factory=list)

    def log(self, message: str, logger: logging.Logger | None = None) -> None:
        self.logs.append(message)
        if logger is not None:
            logger.info(message)


@dataclass
class ParameterSamplingStage:
    parameter_set: ParameterSet
    overrides: Mapping[str, object] | None = None
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        if self.overrides:
            self.parameter_set.override(self.overrides)
            context.log(
                f"[ParameterSampling] applied overrides: {dict(self.overrides)}",
                self.logger,
            )

        sampled = self.parameter_set.sample(seed=context.seed)
        context.sampled_parameters = sampled
        context.log(
            f"[ParameterSampling] seed={context.seed}, sampled={sampled}",
            self.logger,
        )
        return context


@dataclass
class LayoutGenerationStage:
    base_parameters: Mapping[str, object] | None = None
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        merged = self._merge_parameters(context.sampled_parameters)
        factory_params = self._to_factory_params(merged)

        generator = FactoryGenerator(factory_params)
        layout = generator.generate_layout()

        context.resolved_factory_parameters = merged
        context.factory_params = factory_params
        context.factory_generator = generator
        context.layout = layout
        context.log(
            (
                "[LayoutGeneration] "
                f"factory_params={factory_params}, rooms={len(layout)}"
            ),
            self.logger,
        )
        return context

    def _merge_parameters(self, sampled: Mapping[str, float]) -> dict[str, object]:
        merged: dict[str, object] = {}
        if self.base_parameters:
            merged.update(self.base_parameters)
        merged.update(sampled)
        return merged

    def _to_factory_params(self, values: Mapping[str, object]) -> FactoryParams:
        factory_width = float(values["factory_width"])
        factory_depth = float(values["factory_depth"])
        number_of_rooms = int(round(float(values["number_of_rooms"])))
        room_count_range = self._resolve_optional_room_count_range(values)
        if room_count_range is not None:
            minimum, maximum = room_count_range
            number_of_rooms = max(minimum, min(maximum, number_of_rooms))
        corridor_width = float(values["corridor_width"])
        room_height = float(values["room_height"]) if "room_height" in values else None
        room_size_range = self._resolve_room_size_range(values)
        layout_strategy = str(values.get("layout_strategy", "grid"))
        noise = self._resolve_mapping(values, "noise")
        biomes = self._resolve_mapping(values, "biomes")
        columns = self._resolve_mapping(values, "columns")
        beams = self._resolve_mapping(values, "beams")
        machinery = self._resolve_mapping(values, "machinery")
        exterior = self._resolve_mapping(values, "exterior")
        site = self._resolve_mapping(values, "site")
        seed = int(values["seed"]) if "seed" in values and values["seed"] is not None else 0
        return FactoryParams(
            factory_width=factory_width,
            factory_depth=factory_depth,
            number_of_rooms=number_of_rooms,
            room_size_range=room_size_range,
            corridor_width=corridor_width,
            room_height=room_height,
            layout_strategy=layout_strategy,
            noise=noise,
            biomes=biomes,
            columns=columns,
            beams=beams,
            machinery=machinery,
            exterior=exterior,
            site=site,
            seed=seed,
        )

    def _resolve_room_size_range(self, values: Mapping[str, object]) -> tuple[float, float]:
        if "room_size_min" in values and "room_size_max" in values:
            minimum = float(values["room_size_min"])
            maximum = float(values["room_size_max"])
            return (minimum, maximum)

        if "room_size_range" in values:
            raw = values["room_size_range"]
            if isinstance(raw, Mapping):
                if "min" in raw and "max" in raw:
                    return (float(raw["min"]), float(raw["max"]))
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                return (float(raw[0]), float(raw[1]))

        raise KeyError(
            "Missing room size range. Provide room_size_min/room_size_max "
            "or room_size_range."
        )

    def _resolve_optional_room_count_range(
        self,
        values: Mapping[str, object],
    ) -> tuple[int, int] | None:
        if "room_count_range" not in values:
            return None

        raw = values["room_count_range"]
        if isinstance(raw, Mapping):
            if "min" in raw and "max" in raw:
                minimum = int(raw["min"])
                maximum = int(raw["max"])
                if minimum > maximum:
                    raise ValueError("room_count_range min cannot be greater than max.")
                return (minimum, maximum)
            return None

        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            minimum = int(raw[0])
            maximum = int(raw[1])
            if minimum > maximum:
                raise ValueError("room_count_range min cannot be greater than max.")
            return (minimum, maximum)

        return None

    def _resolve_mapping(self, values: Mapping[str, object], key: str) -> dict[str, object]:
        raw = values.get(key)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"'{key}' must be a mapping.")
        return {str(k): v for k, v in raw.items()}


@dataclass
class RoomGenerationStage:
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        generator = context.factory_generator
        if generator is None:
            raise ValueError("Factory generator is missing. Run LayoutGenerationStage first.")

        context.scene = generator.instantiate_rooms()
        context.log(
            f"[RoomGeneration] room_roots={len(context.scene.objects)}",
            self.logger,
        )
        return context


@dataclass
class SceneAssemblyStage:
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        generator = context.factory_generator
        if generator is None:
            raise ValueError("Factory generator is missing. Run LayoutGenerationStage first.")

        context.scene = generator.place_corridors()
        context.scene = generator.place_exterior()
        context.scene = generator.place_site()
        context.scene = generator.place_factory_flow()
        context.scene = generator.place_columns()
        context.scene = generator.place_beams()
        context.scene = generator.place_machinery()
        root_count = len(context.scene.objects) if context.scene is not None else 0
        context.log(
            f"[SceneAssembly] scene_root_objects={root_count}",
            self.logger,
        )
        return context


@dataclass
class InfrastructureStage:
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        generator = context.factory_generator
        if generator is None:
            raise ValueError("Factory generator is missing. Run LayoutGenerationStage first.")
        if context.scene is None:
            raise ValueError("Scene is missing. Run RoomGeneration/SceneAssembly stages first.")

        context.scene = generator.place_infrastructure()
        infra_count = (
            len([obj for obj in context.scene.traverse() if obj.type.startswith("infra_")])
            if context.scene is not None
            else 0
        )
        context.log(
            f"[Infrastructure] objects={infra_count}",
            self.logger,
        )
        return context


@dataclass
class AuxiliaryStage:
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        generator = context.factory_generator
        if generator is None:
            raise ValueError("Factory generator is missing. Run LayoutGenerationStage first.")
        if context.scene is None:
            raise ValueError("Scene is missing. Run RoomGeneration/SceneAssembly stages first.")

        context.scene = generator.place_auxiliary()
        aux_count = (
            len([obj for obj in context.scene.traverse() if obj.type.startswith("aux_")])
            if context.scene is not None
            else 0
        )
        context.log(
            f"[Auxiliary] objects={aux_count}",
            self.logger,
        )
        return context


@dataclass
class ExportStage:
    export_path: str
    exporter: ObjExporter = field(default_factory=ObjExporter)
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.scene is None:
            raise ValueError("Scene is missing. Run RoomGeneration/SceneAssembly stages first.")

        target_path = self._resolve_unique_export_path()
        self.exporter.export(context.scene, str(target_path))
        context.export_path = str(target_path)
        context.log(
            f"[Export] path={target_path}",
            self.logger,
        )
        return context

    def _resolve_unique_export_path(self) -> Path:
        configured = Path(self.export_path)
        unique = _with_suffix_token(configured, _timestamp_token())
        unique.parent.mkdir(parents=True, exist_ok=True)
        return _next_available_path(unique)


@dataclass
class LidarStationPlanningStage:
    settings: Mapping[str, object] | None = None
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.scene is None:
            raise ValueError("Scene is missing. Run ExportStage after scene generation first.")

        generator = LidarSurveyGenerator(self.settings)
        stations = generator.plan_stations(context.scene)
        context.lidar_station_records = [station.to_dict() for station in stations]
        context.log(
            f"[LidarStationPlanning] stations={len(stations)}",
            self.logger,
        )
        return context


@dataclass
class LidarPointCloudStage:
    settings: Mapping[str, object] | None = None
    output_path: str | None = None
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.scene is None:
            raise ValueError("Scene is missing. Run ExportStage after scene generation first.")

        generator = LidarSurveyGenerator(self.settings)
        stations = self._coerce_stations(context.lidar_station_records)
        if not stations:
            stations = generator.plan_stations(context.scene)
            context.lidar_station_records = [station.to_dict() for station in stations]
        stations = self._apply_memory_guard(stations, generator, context)

        circles = generator.generate_scans(context.scene, stations)
        result = LidarSurveyResult(
            stations=stations,
            circles=circles,
            label_map=generator.label_map,
            source_obj_path=context.export_path,
        )

        outputs = self._resolve_output_targets()
        context.lidar_output_dir = outputs["output_dir"]

        if outputs["export_individual_circles"]:
            context.lidar_circle_file_paths = result.write_circle_ply_files(
                output_dir=outputs["circle_output_dir"],
                file_prefix=outputs["circle_file_prefix"],
            )
        else:
            context.lidar_circle_file_paths = []

        if outputs["export_stitched_cloud"]:
            stitched_paths = result.write_split_stitched_ply(
                output_root_dir=outputs["output_dir"],
                combined_path=outputs["stitched_output_path"],
                blind_spot_radius=float(outputs["blind_spot_radius"]),
                fill_blind_spots=bool(outputs["stitched_fill_blind_spots"]),
            )
            context.lidar_stitched_cloud_path = stitched_paths["combined"]
            context.lidar_stitched_cloud_interior_path = stitched_paths["interior"]
            context.lidar_stitched_cloud_exterior_path = stitched_paths["exterior"]
        else:
            context.lidar_stitched_cloud_path = None
            context.lidar_stitched_cloud_interior_path = None
            context.lidar_stitched_cloud_exterior_path = None

        context.lidar_room_stitched_paths = {}
        context.lidar_biome_stitched_paths = {}
        if outputs["export_stitched_by_room"] or outputs["export_stitched_by_biome"]:
            by_room: dict[str, list] = {}
            for circle in circles:
                by_room.setdefault(circle.room_id, []).extend(circle.points)

            if outputs["export_stitched_by_room"]:
                room_dir = Path(str(outputs["room_stitched_output_dir"]))
                room_dir.mkdir(parents=True, exist_ok=True)
                context.lidar_room_stitched_paths = result.write_grouped_stitched_ply(
                    output_dir=str(room_dir),
                    grouped_points=by_room,
                    file_prefix="room",
                    circle_index_base=-1000,
                )

            if outputs["export_stitched_by_biome"]:
                room_to_biome = {
                    spec.room_id: spec.biome.strip().lower()
                    for spec in context.layout
                }
                by_biome: dict[str, list] = {}
                for room_id, points_in_room in by_room.items():
                    biome = room_to_biome.get(room_id, "unknown")
                    by_biome.setdefault(biome, []).extend(points_in_room)
                biome_dir = Path(str(outputs["biome_stitched_output_dir"]))
                biome_dir.mkdir(parents=True, exist_ok=True)
                context.lidar_biome_stitched_paths = result.write_grouped_stitched_ply(
                    output_dir=str(biome_dir),
                    grouped_points=by_biome,
                    file_prefix="biome",
                    circle_index_base=-2000,
                )

        if outputs["export_metadata_json"]:
            context.lidar_dataset_path = result.write_json(outputs["metadata_output_path"])
        else:
            context.lidar_dataset_path = None

        retain_circle_records = _to_bool(
            (self.settings or {}).get("retain_circle_records_in_context", False),
            default=False,
        )
        if retain_circle_records:
            context.lidar_circle_records = [circle.to_dict() for circle in circles]
        else:
            # Keep only lightweight summary to avoid duplicating full point clouds in RAM.
            context.lidar_circle_records = [
                {
                    "station_id": circle.station_id,
                    "room_id": circle.room_id,
                    "point_count": len(circle.points),
                    "center": list(circle.center),
                    "radius": circle.radius,
                }
                for circle in circles
            ]
        context.lidar_labels = dict(result.label_map)
        point_count = sum(len(circle.points) for circle in circles)
        context.log(
            (
                "[LidarPointCloud] "
                f"circles={len(circles)}, points={point_count}, "
                f"circle_files={len(context.lidar_circle_file_paths)}, "
                f"stitched={context.lidar_stitched_cloud_path}, "
                f"stitched_interior={context.lidar_stitched_cloud_interior_path}, "
                f"stitched_exterior={context.lidar_stitched_cloud_exterior_path}, "
                f"room_splits={len(context.lidar_room_stitched_paths)}, "
                f"biome_splits={len(context.lidar_biome_stitched_paths)}, "
                f"metadata={context.lidar_dataset_path}"
            ),
            self.logger,
        )
        return context

    def _apply_memory_guard(
        self,
        stations: list[LidarStation],
        generator: LidarSurveyGenerator,
        context: PipelineContext,
    ) -> list[LidarStation]:
        settings = dict(self.settings or {})
        safe_mode = _to_bool(settings.get("memory_safe_mode", True), default=True)
        if not safe_mode or not stations:
            return stations

        max_total_points = int(settings.get("max_total_points_in_memory", 1_200_000))
        if max_total_points <= 0:
            return stations

        station_count = len(stations)
        point_limit = int(getattr(generator, "points_per_station", 0))
        min_points_per_station = max(
            500,
            int(settings.get("min_points_per_station", 1000)),
        )
        trim_station_count = _to_bool(
            settings.get("trim_station_count_for_memory", False),
            default=False,
        )

        if point_limit <= 0:
            # Unlimited per-station points are unsafe for heavy scans; derive capped budget.
            point_limit = max(min_points_per_station, max_total_points // max(1, station_count))
            generator.points_per_station = point_limit
            context.log(
                (
                    "[LidarGuard] points_per_station was unlimited; "
                    f"auto-capped to {point_limit} (max_total_points_in_memory={max_total_points})"
                ),
                self.logger,
            )

        required_total_points = station_count * point_limit
        if required_total_points <= max_total_points:
            return stations

        # First reduce per-station point limit, preserving station coverage.
        capped_point_limit = max(min_points_per_station, max_total_points // max(1, station_count))
        if capped_point_limit < point_limit:
            generator.points_per_station = capped_point_limit
            point_limit = capped_point_limit
            required_total_points = station_count * point_limit
            context.log(
                (
                    "[LidarGuard] reduced points_per_station to "
                    f"{point_limit} for memory safety "
                    f"(max_total_points_in_memory={max_total_points})"
                ),
                self.logger,
            )

        if required_total_points <= max_total_points:
            return stations

        # If still too large, optionally trim station count.
        if not trim_station_count:
            context.log(
                (
                    "[LidarGuard] keeping all stations to preserve coverage; "
                    "set lidar.trim_station_count_for_memory=true to allow trimming."
                ),
                self.logger,
            )
            return stations

        max_stations_by_budget = max(1, max_total_points // max(1, point_limit))
        if station_count > max_stations_by_budget:
            trimmed = stations[:max_stations_by_budget]
            context.log(
                (
                    "[LidarGuard] reduced station count to "
                    f"{len(trimmed)} (from {station_count}) for memory safety "
                    f"(points_per_station={point_limit}, max_total_points_in_memory={max_total_points})"
                ),
                self.logger,
            )
            return trimmed
        return stations

    def _coerce_stations(self, raw_records: list[dict[str, object]]) -> list[LidarStation]:
        stations: list[LidarStation] = []
        for index, record in enumerate(raw_records, start=1):
            if not isinstance(record, Mapping):
                continue
            try:
                station = LidarStation(
                    id=str(record.get("id", f"lidar_station_{index}")),
                    room_id=str(record["room_id"]),
                    x=float(record["x"]),
                    y=float(record["y"]),
                    z=float(record["z"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            stations.append(station)
        return stations

    def _resolve_output_targets(self) -> dict[str, object]:
        settings = dict(self.settings or {})
        configured_metadata_path = str(
            settings.get(
                "metadata_output_path",
                settings.get("output_path", self.output_path or "out/lidar_scans.json"),
            )
        )
        metadata_path = Path(configured_metadata_path)

        output_dir_value = settings.get("output_dir")
        if output_dir_value is None:
            base_output_dir = Path(str(metadata_path.with_suffix("")))
        else:
            base_output_dir = Path(str(output_dir_value))

        date_folder_enabled = _to_bool(settings.get("date_folder_enabled", True), default=True)
        date_folder_name = str(settings.get("date_folder_name", date.today().isoformat()))
        unique_outputs = _to_bool(settings.get("unique_outputs", True), default=True)
        run_folder_enabled = _to_bool(settings.get("run_folder_enabled", True), default=True)
        run_folder_name = str(settings.get("run_folder_name", _timestamp_token()))
        if date_folder_enabled:
            output_dir_path = base_output_dir / date_folder_name
        else:
            output_dir_path = base_output_dir
        if run_folder_enabled:
            output_dir_path = output_dir_path / run_folder_name
        metadata_output_path = output_dir_path / metadata_path.name

        configured_stitched_path = settings.get("stitched_output_path")
        if configured_stitched_path is None:
            stitched_output_path = output_dir_path / "combined" / "factory_stitched.ply"
        else:
            stitched_name = Path(str(configured_stitched_path)).name
            stitched_output_path = output_dir_path / "combined" / stitched_name

        circle_output_dir = output_dir_path / "circles"
        circle_prefix_base = str(settings.get("circle_file_prefix", "scan_circle"))
        circle_file_prefix = (
            f"{circle_prefix_base}_{run_folder_name}"
            if unique_outputs and not run_folder_enabled
            else circle_prefix_base
        )
        if unique_outputs and not run_folder_enabled:
            metadata_output_path = _next_available_path(metadata_output_path)
            stitched_output_path = _next_available_path(stitched_output_path)

        return {
            "metadata_output_path": str(metadata_output_path),
            "output_dir": str(output_dir_path),
            "circle_output_dir": str(circle_output_dir),
            "stitched_output_path": str(stitched_output_path),
            "circle_file_prefix": circle_file_prefix,
            "export_individual_circles": _to_bool(
                settings.get("export_individual_circles", True),
                default=True,
            ),
            "export_stitched_cloud": _to_bool(
                settings.get("export_stitched_cloud", True),
                default=True,
            ),
            "export_metadata_json": _to_bool(
                settings.get("export_metadata_json", True),
                default=True,
            ),
            "blind_spot_radius": float(settings.get("blind_spot_radius", 0.0)),
            "stitched_fill_blind_spots": _to_bool(
                settings.get("stitched_fill_blind_spots", True),
                default=True,
            ),
            "export_stitched_by_room": _to_bool(
                settings.get("export_stitched_by_room", False),
                default=False,
            ),
            "export_stitched_by_biome": _to_bool(
                settings.get("export_stitched_by_biome", False),
                default=False,
            ),
            "room_stitched_output_dir": str(output_dir_path / "rooms"),
            "biome_stitched_output_dir": str(output_dir_path / "biomes"),
        }


@dataclass
class ScenePipeline:
    stages: list[PipelineStage]
    seed: int | None = 0
    logger: logging.Logger | None = None

    def run(self, context: PipelineContext | None = None) -> PipelineContext:
        current = context or PipelineContext(seed=self.seed)
        if context is not None and current.seed is None:
            current.seed = self.seed

        current.log(
            f"[Pipeline] started with seed={current.seed}",
            self.logger,
        )
        for stage in self.stages:
            current = stage.run(current)
        current.log("[Pipeline] completed", self.logger)
        return current


def build_default_scene_pipeline(
    parameter_set: ParameterSet,
    export_path: str,
    seed: int | None = 0,
    overrides: Mapping[str, object] | None = None,
    base_parameters: Mapping[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> ScenePipeline:
    stages: list[PipelineStage] = [
        ParameterSamplingStage(
            parameter_set=parameter_set,
            overrides=overrides,
            logger=logger,
        ),
        LayoutGenerationStage(
            base_parameters=base_parameters,
            logger=logger,
        ),
        RoomGenerationStage(logger=logger),
        SceneAssemblyStage(logger=logger),
        InfrastructureStage(logger=logger),
        AuxiliaryStage(logger=logger),
        ExportStage(export_path=export_path, logger=logger),
    ]

    lidar_settings: dict[str, object] = {}
    if isinstance(base_parameters, Mapping):
        raw_lidar = base_parameters.get("lidar")
        if isinstance(raw_lidar, Mapping):
            lidar_settings = {str(key): value for key, value in raw_lidar.items()}
    if _to_bool(lidar_settings.get("enabled", False), default=False):
        lidar_output_path = str(lidar_settings.get("output_path", "out/lidar_scans.json"))
        stages.append(
            LidarStationPlanningStage(
                settings=lidar_settings,
                logger=logger,
            )
        )
        stages.append(
            LidarPointCloudStage(
                settings=lidar_settings,
                output_path=lidar_output_path,
                logger=logger,
            )
        )
    return ScenePipeline(stages=stages, seed=seed, logger=logger)
