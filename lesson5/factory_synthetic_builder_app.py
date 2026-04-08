from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset import sanitize_filesystem_path
from factory_showcase_pipeline import (
    RoomInfo,
    build_single_room_from_cloud,
    generate_factory_cloud,
    load_point_cloud_file,
    place_object_files_in_factory_cloud,
    save_combined_cloud_as,
    save_factory_cloud_as,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "factory_demo_runs"
EMBEDDED_MODE = bool(globals().get("EMBEDDED_MODE", False))


def _build_cloud_figure(
    points,
    labels=None,
    *,
    title: str,
    max_points: int = 80000,
    point_size: int = 2,
):
    if points is None or len(points) == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    current_points = points
    current_labels = labels
    if max_points > 0 and len(points) > max_points:
        step = max(1, len(points) // max_points)
        current_points = points[::step]
        if labels is not None:
            current_labels = labels[::step]

    marker = {"size": point_size, "opacity": 0.85}
    if current_labels is not None and len(current_labels) == len(current_points):
        marker["color"] = current_labels
        marker["colorscale"] = "Turbo"
        marker["colorbar"] = {"title": "Метка"}
    else:
        marker["color"] = "#2A7FFF"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=current_points[:, 0],
                y=current_points[:, 1],
                z=current_points[:, 2],
                mode="markers",
                marker=marker,
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
    )
    return fig


def _ensure_state() -> None:
    st.session_state.setdefault("factory_state", None)
    st.session_state.setdefault("placement_state", None)
    st.session_state.setdefault("object_queue", [])


def _rooms_from_metadata(payload: dict) -> tuple[list[RoomInfo], dict[str, int]]:
    rooms: list[RoomInfo] = []
    for item in payload.get("rooms", []):
        try:
            rooms.append(
                RoomInfo(
                    room_id=str(item["room_id"]),
                    biome=str(item.get("biome", "loaded")),
                    center_x=float(item["center_x"]),
                    center_y=float(item["center_y"]),
                    width=float(item["width"]),
                    depth=float(item["depth"]),
                    floor_z=float(item.get("floor_z", 0.0)),
                    ceiling_z=float(item.get("ceiling_z", 6.0)),
                )
            )
        except Exception:
            continue

    lookup_raw = payload.get("room_index_lookup", {})
    room_index_lookup: dict[str, int] = {}
    if isinstance(lookup_raw, dict):
        for key, value in lookup_raw.items():
            try:
                room_index_lookup[str(key)] = int(value)
            except Exception:
                continue

    if not rooms:
        room_index_lookup = {}
    elif not room_index_lookup:
        room_index_lookup = {room.room_id: index for index, room in enumerate(rooms)}

    return rooms, room_index_lookup


def _flatten_detected_rows(detection_by_room: dict[str, list[dict]]) -> pd.DataFrame:
    rows: list[dict] = []
    for room_id, objects in sorted(detection_by_room.items()):
        for item in objects:
            rows.append(
                {
                    "Комната": room_id,
                    "ID объекта": item.get("instance_id"),
                    "Класс": item.get("object_class"),
                    "Точек": item.get("point_count"),
                    "Уверенность": item.get("confidence"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Комната", "ID объекта", "Класс", "Точек", "Уверенность"])
    return pd.DataFrame(rows)


def _objects_table(placement_state) -> pd.DataFrame:
    rows: list[dict] = []
    for obj in placement_state.placed_objects:
        rows.append(
            {
                "ID объекта": obj.instance_id,
                "Класс": obj.object_class,
                "Комната": obj.room_id,
                "Точек": int(len(obj.points)),
                "Файл": obj.source_file,
            }
        )
    return pd.DataFrame(rows)


def _queue_table(items: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for index, item in enumerate(items, start=1):
        rows.append(
            {
                "№": index,
                "Файл облака": item.get("file_path", ""),
                "Тип объекта": item.get("class_name", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["№", "Файл облака", "Тип объекта"])
    return pd.DataFrame(rows)


if not EMBEDDED_MODE:
    st.set_page_config(
        page_title="Конструктор синтетических облаков",
        page_icon="",
        layout="wide",
    )

_ensure_state()

st.title("Конструктор синтетических облаков точек")
st.caption(
    "Отдельная часть для искусственного создания данных: генерация завода, размещение объектов и сохранение облаков."
)

with st.sidebar:
    st.subheader("Пути")
    output_root = st.text_input("Папка результатов", value=str(DEFAULT_OUTPUT_DIR))

st.markdown("## 1) Генерация или загрузка завода")

tab_gen, tab_load = st.tabs(["Сгенерировать завод", "Загрузить готовый завод"])

with tab_gen:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        gen_seed = st.number_input("Seed", min_value=1, value=42, step=1)
        factory_width = st.number_input("Ширина завода (м)", min_value=40.0, value=120.0, step=5.0)
        factory_depth = st.number_input("Глубина завода (м)", min_value=40.0, value=90.0, step=5.0)
    with col_b:
        room_count = st.number_input("Количество комнат", min_value=2, value=9, step=1)
        room_height = st.number_input("Высота комнат (м)", min_value=3.0, value=6.0, step=0.5)
        use_lasersensing = st.checkbox("Использовать laserSensing для расстановки лидаров", value=False)
    with col_c:
        lidar_density = st.slider(
            "Плотность облака LiDAR",
            min_value=0.25,
            max_value=3.0,
            value=1.0,
            step=0.25,
        )
        st.write("1.0 = базовая плотность, >1.0 плотнее, <1.0 реже")

    if st.button("Сгенерировать завод и облако точек"):
        with st.spinner("Генерация завода и LiDAR-облака..."):
            result = generate_factory_cloud(
                output_root=output_root,
                seed=int(gen_seed),
                factory_width=float(factory_width),
                factory_depth=float(factory_depth),
                room_count=int(room_count),
                room_height=float(room_height),
                lidar_density=float(lidar_density),
                use_lasersensing=bool(use_lasersensing),
            )
        st.session_state["factory_state"] = {
            "source": "generated",
            "points": result.factory_points,
            "labels": result.factory_labels,
            "room_indices": result.factory_room_indices,
            "rooms": result.rooms,
            "room_index_lookup": result.room_index_lookup,
            "cloud_path": result.factory_cloud_path,
            "metadata_path": result.metadata_path,
            "output_dir": result.output_dir,
            "notes": result.notes,
        }
        st.success("Завод успешно сгенерирован.")

with tab_load:
    cloud_path = st.text_input("Путь к облаку точек завода (.ply)", value="")
    metadata_path = st.text_input(
        "Путь к метаданным завода (необязательно, JSON)",
        value="",
    )
    if st.button("Загрузить готовый завод"):
        with st.spinner("Загрузка облака завода..."):
            cloud_path_clean = sanitize_filesystem_path(cloud_path)
            metadata_path_clean = sanitize_filesystem_path(metadata_path)
            points, labels = load_point_cloud_file(cloud_path_clean)
            room_indices = labels.copy()
            rooms: list[RoomInfo] = []
            room_lookup: dict[str, int] = {}
            if metadata_path_clean:
                payload = json.loads(Path(metadata_path_clean).read_text(encoding="utf-8"))
                rooms, room_lookup = _rooms_from_metadata(payload)
            if not rooms:
                rooms = build_single_room_from_cloud(points)
                room_lookup = {rooms[0].room_id: 0}
                room_indices = (room_indices * 0).astype(int)

        st.session_state["factory_state"] = {
            "source": "loaded",
            "points": points,
            "labels": labels,
            "room_indices": room_indices,
            "rooms": rooms,
            "room_index_lookup": room_lookup,
            "cloud_path": cloud_path_clean,
            "metadata_path": metadata_path_clean or "",
            "output_dir": output_root,
            "notes": [],
        }
        st.success("Готовый завод загружен.")

factory_state = st.session_state.get("factory_state")
if factory_state is not None:
    st.write(f"Источник завода: **{factory_state['source']}**")
    st.write(f"Облако: `{factory_state['cloud_path']}`")
    if factory_state.get("metadata_path"):
        st.write(f"Метаданные: `{factory_state['metadata_path']}`")

    for note in factory_state.get("notes", []):
        st.warning(note)

    c1, c2, c3 = st.columns(3)
    c1.metric("Точек в заводе", int(len(factory_state["points"])))
    c2.metric("Комнат", int(len(factory_state["rooms"])))
    c3.metric("Диапазон меток", f"{int(factory_state['labels'].min())}..{int(factory_state['labels'].max())}")

    st.plotly_chart(
        _build_cloud_figure(
            factory_state["points"],
            factory_state["labels"],
            title="Облако точек завода",
            max_points=90000,
            point_size=2,
        ),
        use_container_width=True,
    )

    st.markdown("### Сохранение облака завода")
    default_factory_export = str(Path(output_root) / "factory_cloud_saved.ply")
    factory_export_path = st.text_input(
        "Куда сохранить облако завода (.ply)",
        value=default_factory_export,
        key="factory_export_path",
    )
    if st.button("Сохранить облако завода как..."):
        try:
            saved_path = save_factory_cloud_as(
                path=factory_export_path,
                points=factory_state["points"],
                labels=factory_state["labels"],
                room_indices=factory_state["room_indices"],
            )
        except Exception as export_error:
            st.error(f"Ошибка сохранения облака завода: {export_error}")
        else:
            st.success(f"Облако завода сохранено: {saved_path}")
else:
    st.info("Сначала сгенерируйте завод или загрузите готовое облако.")

st.markdown("## 2) Размещение облаков объектов в заводе")
if factory_state is None:
    st.warning("Блок размещения доступен после подготовки облака завода.")
else:
    st.caption("Добавляйте объекты по одному: укажите конкретный `.ply`, добавьте в очередь и выполните размещение.")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        one_object_path = st.text_input("Путь к облаку объекта (.ply)", value="")
        one_object_type = st.text_input("Тип объекта (если пусто, возьмётся имя папки)", value="")
    with col_q2:
        placement_seed = st.number_input("Seed размещения", min_value=1, value=42, step=1)
        max_points_per_object = st.number_input(
            "Макс. точек на один объект",
            min_value=1000,
            value=14000,
            step=1000,
        )
    with col_q3:
        global_scale = st.slider(
            "Масштаб размещаемых объектов",
            min_value=0.4,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )

    add_col, clear_col, drop_col = st.columns(3)
    with add_col:
        if st.button("Добавить объект в очередь"):
            object_path_clean = sanitize_filesystem_path(one_object_path)
            candidate = Path(object_path_clean)
            if not object_path_clean:
                st.error("Укажите путь к файлу объекта.")
            elif not candidate.exists():
                st.error("Файл не найден.")
            elif candidate.suffix.lower() != ".ply":
                st.error("Нужен файл формата .ply")
            else:
                class_name = one_object_type.strip() or candidate.parent.name or candidate.stem
                st.session_state["object_queue"].append(
                    {"file_path": str(candidate), "class_name": class_name}
                )
                st.success("Объект добавлен в очередь.")
    with clear_col:
        if st.button("Очистить очередь"):
            st.session_state["object_queue"] = []
            st.info("Очередь очищена.")
    with drop_col:
        queue_len = len(st.session_state.get("object_queue", []))
        remove_index = st.number_input(
            "Удалить № из очереди",
            min_value=1,
            max_value=max(1, queue_len),
            value=1,
            step=1,
            key="remove_queue_index_builder",
        )
        if st.button("Удалить из очереди"):
            queue = st.session_state.get("object_queue", [])
            if not queue:
                st.warning("Очередь пуста.")
            else:
                idx = int(remove_index) - 1
                if 0 <= idx < len(queue):
                    removed = queue.pop(idx)
                    st.success(f"Удалён: {removed.get('file_path')}")

    st.dataframe(_queue_table(st.session_state.get("object_queue", [])), use_container_width=True, hide_index=True)

    if st.button("Разместить объекты из очереди в заводе"):
        queue = st.session_state.get("object_queue", [])
        if not queue:
            st.error("Очередь пуста. Добавьте хотя бы один объект.")
        else:
            with st.spinner("Размещение облаков объектов..."):
                placement = place_object_files_in_factory_cloud(
                    factory_points=factory_state["points"],
                    factory_labels=factory_state["labels"],
                    factory_room_indices=factory_state["room_indices"],
                    rooms=factory_state["rooms"],
                    room_index_lookup=factory_state["room_index_lookup"],
                    object_items=queue,
                    output_root=output_root,
                    max_points_per_object=int(max_points_per_object),
                    seed=int(placement_seed),
                    global_scale=float(global_scale),
                )
            st.session_state["placement_state"] = placement
            st.success("Размещение завершено.")

placement_state = st.session_state.get("placement_state")
if placement_state is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Размещено объектов", int(len(placement_state.placed_objects)))
    m2.metric("Точек в комбинированном облаке", int(len(placement_state.combined_points)))
    m3.metric("Классов объектов", int(len(placement_state.dataset_classes)))

    st.write(f"Комбинированное облако (автосохранение): `{placement_state.combined_ply_path}`")
    st.write(f"Метаданные размещения: `{placement_state.metadata_path}`")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(
            _build_cloud_figure(
                placement_state.combined_points,
                placement_state.combined_source,
                title="Комбинированное облако (0=завод, 1=объекты)",
                max_points=100000,
                point_size=2,
            ),
            use_container_width=True,
        )
    with right:
        st.dataframe(_flatten_detected_rows(placement_state.detection_by_room), use_container_width=True, hide_index=True)

    with st.expander("Список всех размещённых объектов"):
        st.dataframe(_objects_table(placement_state), use_container_width=True, hide_index=True)

    st.markdown("### Сохранение комбинированного облака")
    default_combined_export = str(Path(output_root) / "combined_cloud_saved.ply")
    combined_export_path = st.text_input(
        "Куда сохранить комбинированное облако (.ply)",
        value=default_combined_export,
        key="combined_export_path",
    )
    if st.button("Сохранить комбинированное облако как..."):
        try:
            saved_path = save_combined_cloud_as(
                path=combined_export_path,
                points=placement_state.combined_points,
                source=placement_state.combined_source,
                factory_label=placement_state.combined_factory_label,
                object_type=placement_state.combined_object_type,
                object_instance=placement_state.combined_object_instance,
                part_label=placement_state.combined_part_label,
                room_index=placement_state.combined_room_index,
            )
        except Exception as export_error:
            st.error(f"Ошибка сохранения комбинированного облака: {export_error}")
        else:
            st.success(f"Комбинированное облако сохранено: {saved_path}")
