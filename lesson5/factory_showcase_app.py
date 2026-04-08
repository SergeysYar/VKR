from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset import sanitize_filesystem_path
from engineering_classifier import find_latest_checkpoint
from factory_showcase_pipeline import (
    RoomInfo,
    build_single_room_from_cloud,
    classify_object_parts,
    finetune_recognition_model,
    finetune_object_model,
    generate_factory_cloud,
    load_point_cloud_file,
    place_object_files_in_factory_cloud,
    recognize_cloud_with_model,
    reconstruct_object_surfaces,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = Path(r"D:\_Датасеты\МФТИ DS2-20260408T080031Z-3-002\МФТИ DS2")
DEFAULT_OUTPUT_DIR = BASE_DIR / "factory_demo_runs"
DEFAULT_WEIGHTS_BASE = Path(r"D:\vesa")
DEFAULT_RECOGNITION_WEIGHTS_DIR = DEFAULT_WEIGHTS_BASE / "recognition"
DEFAULT_SEGMENTATION_WEIGHTS_DIR = DEFAULT_WEIGHTS_BASE / "segmentation"


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
    st.session_state.setdefault("recognition_state", None)
    st.session_state.setdefault("recognition_finetune_state", None)
    st.session_state.setdefault("segmentation_finetune_state", None)
    st.session_state.setdefault("segmentations", {})
    st.session_state.setdefault("reconstructions", {})


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


def _class_hist_table(hist: dict[int, int]) -> pd.DataFrame:
    rows = [{"Метка": int(label), "Точек": int(count)} for label, count in sorted(hist.items())]
    if not rows:
        return pd.DataFrame(columns=["Метка", "Точек"])
    return pd.DataFrame(rows)


def _find_auto_weights(weights_root: Path, object_class: str) -> str:
    object_folder = weights_root / object_class / "checkpoints"
    latest = find_latest_checkpoint([object_folder])
    return latest or ""


def _format_instance_option(obj) -> str:
    return f"{obj.instance_id} | класс={obj.object_class} | комната={obj.room_id}"


st.set_page_config(
    page_title="ПО классификации инженерных объектов",
    page_icon="",
    layout="wide",
)

_ensure_state()

st.title("Программное обеспечение для классификации инженерных объектов из облака точек")
st.caption(
    "Единый русскоязычный интерфейс: генерация/загрузка завода, размещение объектов, распознавание, дообучение, сегментация и восстановление поверхности."
)
st.info(
    "Доступны отдельные приложения: "
    "`factory_tabs_app.py` (единое приложение с вкладками), "
    "`factory_synthetic_builder_app.py` (только генерация/размещение) и "
    "`factory_analysis_app.py` (только анализ готовых облаков)."
)

with st.sidebar:
    st.subheader("Общие пути")
    output_root = st.text_input("Папка результатов", value=str(DEFAULT_OUTPUT_DIR))
    recognition_weights_root = st.text_input(
        "Папка весов распознавания",
        value=str(DEFAULT_RECOGNITION_WEIGHTS_DIR),
    )
    segmentation_weights_root = st.text_input(
        "Папка весов сегментации",
        value=str(DEFAULT_SEGMENTATION_WEIGHTS_DIR),
    )

st.markdown("## 1) Блок генерации или загрузки завода")

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
        st.write("Сформируется облако точек завода")
        st.write("1.0 = базовая плотность, >1.0 плотнее, <1.0 реже")
        st.write("Сохранение: `factory_cloud.ply` + метаданные")

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
else:
    st.info("Сначала сгенерируйте завод или загрузите готовое облако.")

st.markdown("## 2) Размещение облаков точек объектов в заводе")
if factory_state is None:
    st.warning("Блок размещения доступен после подготовки облака завода.")
else:
    st.caption("Добавляйте объекты по одному: укажите конкретный `.ply`, затем сформируйте очередь и выполните размещение.")
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
        st.write("Результат: комбинированное облако + список объектов по комнатам")

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
            key="remove_queue_index",
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

    st.write(f"Комбинированное облако: `{placement_state.combined_ply_path}`")
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

st.markdown("## 3) Блок модели распознавания")
st.caption("Используется отдельная модель распознавания и отдельные веса от сегментации.")

rec_cloud_mode = st.radio(
    "Источник облака для распознавания",
    options=["Облако завода", "Комбинированное облако (после размещения)"],
    horizontal=True,
)
recognition_model_name = st.text_input(
    "Имя модели распознавания (папка весов)",
    value="factory_recognition",
)
auto_recognition_ckpt = _find_auto_weights(
    Path(recognition_weights_root),
    recognition_model_name.strip() or "factory_recognition",
)
if auto_recognition_ckpt:
    st.caption(f"Автоматически найден последний вес распознавания: `{auto_recognition_ckpt}`")
recognition_ckpt = st.text_input(
    "Путь к весам модели распознавания (.ckpt, можно оставить пустым для авто)",
    value="",
)
recognition_points = st.number_input("Точек на инференс", min_value=256, value=4096, step=256)

selected_points = None
if rec_cloud_mode.startswith("Комбинированное") and placement_state is not None:
    selected_points = placement_state.combined_points
elif factory_state is not None:
    selected_points = factory_state["points"]

if st.button("Запустить распознавание объектов в облаке"):
    recognition_ckpt_path = recognition_ckpt.strip() or auto_recognition_ckpt
    if selected_points is None:
        st.error("Нет облака для распознавания.")
    elif not recognition_ckpt_path:
        st.error("Укажите путь к .ckpt модели распознавания или обучите модель в блоке ниже.")
    else:
        with st.spinner("Распознавание выполняется..."):
            recognition = recognize_cloud_with_model(
                points=selected_points,
                checkpoint_path=recognition_ckpt_path,
                output_dir=Path(output_root) / "recognition",
                num_points=int(recognition_points),
            )
        st.session_state["recognition_state"] = recognition
        st.success("Распознавание завершено.")

recognition_state = st.session_state.get("recognition_state")
if recognition_state is not None:
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Средняя уверенность", f"{recognition_state['mean_confidence']:.3f}")
        st.dataframe(_class_hist_table(recognition_state["histogram"]), hide_index=True, use_container_width=True)
        if recognition_state.get("saved_path"):
            st.write(f"Результат сохранён: `{recognition_state['saved_path']}`")
    with r2:
        st.plotly_chart(
            _build_cloud_figure(
                recognition_state["points"],
                recognition_state["predicted_labels"],
                title="Распознанные метки в облаке",
                max_points=80000,
                point_size=2,
            ),
            use_container_width=True,
        )

st.markdown("### Дообучение модели распознавания объектов")
st.caption("Формат датасета распознавания: каждая папка = отдельный тип объекта, внутри папки — варианты поворотов/поз.")
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    train_rec_dataset_root = st.text_input(
        "Папка с наборами объектов (каждая папка = один тип)",
        value=str(DEFAULT_DATASET_DIR),
        key="train_rec_dataset_root",
    )
    train_rec_model_name = st.text_input(
        "Имя модели распознавания (папка весов)",
        value="factory_recognition",
        key="train_rec_model_name",
    )
with col_t2:
    train_rec_num_classes = st.number_input("Классов", min_value=2, value=13, step=1, key="train_rec_num_classes")
    train_rec_epochs = st.number_input("Эпох", min_value=1, value=20, step=1, key="train_rec_epochs")
with col_t3:
    train_rec_batch = st.number_input("Batch size", min_value=1, value=4, step=1, key="train_rec_batch")
    train_rec_lr = st.number_input(
        "Learning rate",
        min_value=0.00001,
        value=0.001,
        step=0.0001,
        format="%.5f",
        key="train_rec_lr",
    )

if st.button("Запустить дообучение распознавания"):
    logs_box = st.empty()
    logs: list[str] = []

    def _status_rec(msg: str) -> None:
        logs.append(msg)
        logs_box.code("\n".join(logs[-14:]))

    with st.spinner("Идёт дообучение модели распознавания..."):
        try:
            result = finetune_recognition_model(
                dataset_root=train_rec_dataset_root,
                model_name=train_rec_model_name,
                weights_root=recognition_weights_root,
                num_classes=int(train_rec_num_classes),
                num_points=4096,
                batch_size=int(train_rec_batch),
                max_epochs=int(train_rec_epochs),
                learning_rate=float(train_rec_lr),
                seed=42,
                status_cb=_status_rec,
            )
        except Exception as train_error:
            st.session_state.pop("recognition_finetune_state", None)
            st.error(f"Ошибка дообучения распознавания: {train_error}")
            logs_box.code("\n".join(logs[-20:]))
        else:
            st.session_state["recognition_finetune_state"] = result
            st.success("Дообучение модели распознавания завершено.")

recognition_finetune_state = st.session_state.get("recognition_finetune_state")
if recognition_finetune_state is not None:
    requested_num_classes = recognition_finetune_state.get("requested_num_classes")
    effective_num_classes = recognition_finetune_state.get("effective_num_classes")
    label_space = recognition_finetune_state.get("label_space") or {}
    if label_space:
        if recognition_finetune_state.get("label_mode") == "folder_name":
            classes = label_space.get("class_names") or []
            preview = ", ".join(str(name) for name in classes[:12])
            if len(classes) > 12:
                preview += ", ..."
            st.caption(
                "Обнаруженные типы объектов (папки): "
                f"{label_space.get('class_count')} шт., "
                f"файлов: {label_space.get('file_count')}."
            )
            if preview:
                st.write(f"Классы: {preview}")
        else:
            st.caption(
                "Обнаруженные метки в датасете: "
                f"{label_space.get('min_label')}..{label_space.get('max_label')} "
                f"(уникальных: {label_space.get('unique_count')}, "
                f"файлов: {label_space.get('labeled_files')}/{label_space.get('file_count')})."
            )
    if recognition_finetune_state.get("num_classes_auto_adjusted"):
        st.warning(
            "Количество классов было автоматически увеличено: "
            f"{requested_num_classes} -> {effective_num_classes}, "
            f"так как по меткам требуется минимум {label_space.get('required_num_classes')}."
        )
    st.write(
        {
            "model_name": recognition_finetune_state.get("object_name"),
            "weights_root": recognition_finetune_state.get("weights_root"),
            "requested_num_classes": requested_num_classes,
            "effective_num_classes": effective_num_classes,
            "best_checkpoint": recognition_finetune_state.get("best_checkpoint"),
            "test_metrics": recognition_finetune_state.get("test_metrics"),
        }
    )

st.markdown("## 4) Сегментация распознанного объекта")
st.caption("Используется отдельная модель сегментации и отдельные веса.")

st.markdown("### Дообучение модели сегментации по типу объекта")
seg_t1, seg_t2, seg_t3 = st.columns(3)
with seg_t1:
    train_seg_dataset_root = st.text_input(
        "Папка с облаками точек этого объекта для сегментации",
        value=str(DEFAULT_DATASET_DIR),
        key="train_seg_dataset_root",
    )
    train_seg_object_name = st.text_input(
        "Имя типа объекта (папка весов сегментации)",
        value="valve",
        key="train_seg_object_name",
    )
with seg_t2:
    train_seg_num_classes = st.number_input("Классов", min_value=2, value=13, step=1, key="train_seg_num_classes")
    train_seg_epochs = st.number_input("Эпох", min_value=1, value=20, step=1, key="train_seg_epochs")
with seg_t3:
    train_seg_batch = st.number_input("Batch size", min_value=1, value=4, step=1, key="train_seg_batch")
    train_seg_lr = st.number_input(
        "Learning rate",
        min_value=0.00001,
        value=0.001,
        step=0.0001,
        format="%.5f",
        key="train_seg_lr",
    )

if st.button("Запустить дообучение сегментации"):
    logs_box = st.empty()
    logs: list[str] = []

    def _status_seg(msg: str) -> None:
        logs.append(msg)
        logs_box.code("\n".join(logs[-14:]))

    with st.spinner("Идёт дообучение модели сегментации..."):
        try:
            result = finetune_object_model(
                dataset_root=train_seg_dataset_root,
                object_name=train_seg_object_name,
                weights_root=segmentation_weights_root,
                num_classes=int(train_seg_num_classes),
                num_points=4096,
                batch_size=int(train_seg_batch),
                max_epochs=int(train_seg_epochs),
                learning_rate=float(train_seg_lr),
                seed=42,
                status_cb=_status_seg,
            )
        except Exception as train_error:
            st.session_state.pop("segmentation_finetune_state", None)
            st.error(f"Ошибка дообучения сегментации: {train_error}")
            logs_box.code("\n".join(logs[-20:]))
        else:
            st.session_state["segmentation_finetune_state"] = result
            st.success("Дообучение модели сегментации завершено.")

segmentation_finetune_state = st.session_state.get("segmentation_finetune_state")
if segmentation_finetune_state is not None:
    requested_num_classes = segmentation_finetune_state.get("requested_num_classes")
    effective_num_classes = segmentation_finetune_state.get("effective_num_classes")
    label_space = segmentation_finetune_state.get("label_space") or {}
    if label_space:
        st.caption(
            "Обнаруженные метки в датасете сегментации: "
            f"{label_space.get('min_label')}..{label_space.get('max_label')} "
            f"(уникальных: {label_space.get('unique_count')}, "
            f"файлов: {label_space.get('labeled_files')}/{label_space.get('file_count')})."
        )
    if segmentation_finetune_state.get("num_classes_auto_adjusted"):
        st.warning(
            "Количество классов сегментации было автоматически увеличено: "
            f"{requested_num_classes} -> {effective_num_classes}, "
            f"так как по меткам требуется минимум {label_space.get('required_num_classes')}."
        )
    st.write(
        {
            "object_name": segmentation_finetune_state.get("object_name"),
            "weights_root": segmentation_finetune_state.get("weights_root"),
            "requested_num_classes": requested_num_classes,
            "effective_num_classes": effective_num_classes,
            "best_checkpoint": segmentation_finetune_state.get("best_checkpoint"),
            "test_metrics": segmentation_finetune_state.get("test_metrics"),
        }
    )

if placement_state is None:
    st.warning("Сегментация по объектам доступна после шага размещения объектов.")
else:
    selected_option = st.selectbox(
        "Выберите объект для сегментации",
        options=[_format_instance_option(obj) for obj in placement_state.placed_objects],
    )
    selected_id = selected_option.split("|", maxsplit=1)[0].strip()
    selected_obj = next(obj for obj in placement_state.placed_objects if obj.instance_id == selected_id)

    auto_ckpt = _find_auto_weights(Path(segmentation_weights_root), selected_obj.object_class)
    ckpt_input = st.text_input(
        "Вес сегментации (.ckpt) для выбранного типа объекта",
        value=auto_ckpt,
    )
    if auto_ckpt:
        st.caption(f"Автоматически найдено в весах сегментации: `{auto_ckpt}`")

    if st.button("Запустить сегментацию выбранного объекта"):
        with st.spinner("Сегментация выполняется..."):
            segmentation = classify_object_parts(
                obj=selected_obj,
                checkpoint_path=ckpt_input.strip() or None,
                num_points=4096,
            )
        st.session_state["segmentations"][selected_obj.instance_id] = segmentation
        st.success("Сегментация завершена.")

    segmentation = st.session_state.get("segmentations", {}).get(selected_obj.instance_id)
    if segmentation is not None:
        if segmentation.notes:
            for note in segmentation.notes:
                st.warning(note)
        s1, s2 = st.columns([1, 2])
        with s1:
            st.write(f"Источник сегментации: **{segmentation.source}**")
            st.dataframe(_class_hist_table(segmentation.histogram), hide_index=True, use_container_width=True)
        with s2:
            st.plotly_chart(
                _build_cloud_figure(
                    segmentation.points,
                    segmentation.labels,
                    title=f"Сегментация объекта {selected_obj.instance_id}",
                    max_points=70000,
                    point_size=3,
                ),
                use_container_width=True,
            )

st.markdown("## 5) Восстановление поверхности и настройки")
if placement_state is None:
    st.info("Шаг восстановления будет доступен после сегментации объекта.")
else:
    selected_option_recon = st.selectbox(
        "Объект для восстановления",
        options=[_format_instance_option(obj) for obj in placement_state.placed_objects],
        key="recon_object_select",
    )
    recon_obj_id = selected_option_recon.split("|", maxsplit=1)[0].strip()
    recon_seg = st.session_state.get("segmentations", {}).get(recon_obj_id)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        min_points_part = st.number_input("Минимум точек на часть", min_value=10, value=45, step=5)
    with rc2:
        prefer_surface_module = st.checkbox("Использовать SurfaceReconstructor при возможности", value=False)
    with rc3:
        recon_subdir = st.text_input("Подпапка результатов восстановления", value="reconstruction")

    if st.button("Запустить восстановление поверхности"):
        if recon_seg is None:
            st.error("Сначала выполните сегментацию выбранного объекта.")
        else:
            target_dir = Path(output_root) / recon_subdir / recon_obj_id
            with st.spinner("Выполняется восстановление поверхности..."):
                recon = reconstruct_object_surfaces(
                    object_id=recon_obj_id,
                    classification=recon_seg,
                    output_dir=target_dir,
                    min_points_per_part=int(min_points_part),
                    prefer_surface_module=bool(prefer_surface_module),
                    checkpoint_path=None,
                )
            st.session_state["reconstructions"][recon_obj_id] = recon
            st.success("Восстановление завершено.")

    recon_result = st.session_state.get("reconstructions", {}).get(recon_obj_id)
    if recon_result is not None:
        st.write(f"Backend восстановления: **{recon_result.backend}**")
        if recon_result.notes:
            for note in recon_result.notes:
                st.warning(note)

        if recon_result.part_summaries:
            st.dataframe(pd.DataFrame(recon_result.part_summaries), hide_index=True, use_container_width=True)

        if len(recon_result.combined_points) > 0:
            st.plotly_chart(
                _build_cloud_figure(
                    recon_result.combined_points,
                    title=f"Восстановленная поверхность: {recon_obj_id}",
                    max_points=90000,
                    point_size=2,
                ),
                use_container_width=True,
            )

        if recon_result.generated_files:
            st.markdown("Сгенерированные файлы")
            st.code("\n".join(recon_result.generated_files))
