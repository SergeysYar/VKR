from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset import sanitize_filesystem_path
from engineering_classifier import find_latest_checkpoint
from factory_showcase_pipeline import (
    build_classification_from_labeled_cloud,
    build_virtual_object,
    classify_object_parts,
    export_reconstruction_obj,
    finetune_object_model,
    finetune_recognition_model,
    load_point_cloud_file,
    recognize_cloud_with_model,
    reconstruct_object_surfaces,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = Path(r"D:\_Датасеты\МФТИ DS2-20260408T080031Z-3-002\МФТИ DS2")
DEFAULT_OUTPUT_DIR = BASE_DIR / "factory_demo_runs"
DEFAULT_WEIGHTS_BASE = Path(r"D:\vesa")
DEFAULT_RECOGNITION_WEIGHTS_DIR = DEFAULT_WEIGHTS_BASE / "recognition"
DEFAULT_SEGMENTATION_WEIGHTS_DIR = DEFAULT_WEIGHTS_BASE / "segmentation"
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


def _load_obj_preview_mesh(
    obj_path: str | Path,
    *,
    max_faces: int = 120000,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(obj_path)
    if not path.exists():
        raise FileNotFoundError(f"OBJ не найден: {path}")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        continue
                continue
            if not line.startswith("f "):
                continue
            tokens = line.strip().split()[1:]
            if len(tokens) < 3:
                continue
            indices: list[int] = []
            for token in tokens:
                idx_token = token.split("/", maxsplit=1)[0].strip()
                if not idx_token:
                    continue
                try:
                    idx = int(idx_token)
                except ValueError:
                    continue
                if idx < 0:
                    idx = len(vertices) + idx + 1
                zero_based = idx - 1
                if zero_based >= 0:
                    indices.append(zero_based)
            if len(indices) < 3:
                continue
            base = indices[0]
            for i in range(1, len(indices) - 1):
                faces.append((base, indices[i], indices[i + 1]))

    if not vertices:
        raise ValueError("OBJ не содержит вершин.")

    vertices_np = np.asarray(vertices, dtype=np.float32)
    valid_faces = [face for face in faces if max(face) < len(vertices_np)]
    faces_np = np.asarray(valid_faces, dtype=np.int32) if valid_faces else np.zeros((0, 3), dtype=np.int32)
    if max_faces > 0 and len(faces_np) > max_faces:
        step = max(1, len(faces_np) // max_faces)
        faces_np = faces_np[::step]
    return vertices_np, faces_np


def _build_obj_figure(
    obj_path: str | Path,
    *,
    title: str,
) -> tuple[go.Figure, int, int]:
    vertices, faces = _load_obj_preview_mesh(obj_path)
    if len(faces) == 0:
        fig = _build_cloud_figure(vertices, None, title=title, max_points=100000, point_size=2)
        return fig, int(len(vertices)), 0

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#9CAEC4",
                opacity=0.95,
                flatshading=True,
                lighting={
                    "ambient": 0.45,
                    "diffuse": 0.9,
                    "specular": 0.2,
                    "roughness": 0.6,
                },
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
    return fig, int(len(vertices)), int(len(faces))


def _class_hist_table(hist: dict[int, int]) -> pd.DataFrame:
    rows = [{"Метка": int(label), "Точек": int(count)} for label, count in sorted(hist.items())]
    if not rows:
        return pd.DataFrame(columns=["Метка", "Точек"])
    return pd.DataFrame(rows)


def _find_auto_weights(weights_root: Path, object_class: str) -> str:
    object_folder = weights_root / object_class / "checkpoints"
    latest = find_latest_checkpoint([object_folder])
    return latest or ""


def _save_uploaded_input_file(
    uploaded_file,
    *,
    output_root: str | Path,
    prefix: str,
    fallback_suffix: str | None = None,
) -> str:
    if uploaded_file is None:
        return ""

    upload_dir = Path(output_root) / "uploaded_inputs"
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(str(getattr(uploaded_file, "name", "") or "uploaded_file")).name
    safe_name = "".join(
        char if (char.isalnum() or char in {"_", "-", "."}) else "_"
        for char in original_name
    ) or "uploaded_file"

    suffix = Path(safe_name).suffix
    if not suffix and fallback_suffix:
        safe_name = f"{safe_name}{fallback_suffix}"

    target_path = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prefix}_{safe_name}"
    target_path.write_bytes(uploaded_file.getbuffer())
    return str(target_path.resolve())


def _resolve_file_input(
    *,
    path_value: str,
    uploaded_file,
    output_root: str | Path,
    prefix: str,
    session_key: str | None = None,
    fallback_suffix: str | None = None,
) -> str:
    if uploaded_file is not None:
        saved_path = _save_uploaded_input_file(
            uploaded_file,
            output_root=output_root,
            prefix=prefix,
            fallback_suffix=fallback_suffix,
        )
        if session_key:
            st.session_state[f"{session_key}__resolved_path"] = saved_path
        return saved_path
    return sanitize_filesystem_path(path_value)


def _ensure_state() -> None:
    st.session_state.setdefault("analysis_cloud_state", None)
    st.session_state.setdefault("recognition_state", None)
    st.session_state.setdefault("recognition_finetune_state", None)
    st.session_state.setdefault("segmentation_finetune_state", None)
    st.session_state.setdefault("segmentation_state", None)
    st.session_state.setdefault("reconstruction_state", None)
    st.session_state.setdefault("demo_mode_state", None)


if not EMBEDDED_MODE:
    st.set_page_config(
        page_title="Анализ готовых облаков",
        page_icon="",
        layout="wide",
    )

_ensure_state()

st.title("Аналитика готовых облаков точек")

with st.sidebar:
    st.subheader("Пути")
    output_root = st.text_input("Папка результатов", value=str(DEFAULT_OUTPUT_DIR))
    recognition_weights_root = st.text_input(
        "Папка весов распознавания",
        value=str(DEFAULT_RECOGNITION_WEIGHTS_DIR),
    )
    segmentation_weights_root = st.text_input(
        "Папка весов сегментации",
        value=str(DEFAULT_SEGMENTATION_WEIGHTS_DIR),
    )

st.markdown("## Демо-режим идеальной работы")
demo_mode_enabled = st.checkbox("Включить демо-режим", value=False, key="demo_mode_enabled")

if demo_mode_enabled:
    demo_cloud_path = st.text_input(
        "Путь к заранее размеченному облаку (.ply)",
        value="",
        key="demo_cloud_path",
    )
    uploaded_demo_cloud = st.file_uploader(
        "Или выберите облако для демо через диалог",
        type=["ply"],
        key="demo_cloud_upload",
    )
    demo_object_class = st.text_input(
        "Тип объекта для демо",
        value="valve",
        key="demo_object_class",
    )
    demo_min_points_part = st.number_input(
        "Мин. точек на сегмент (демо)",
        min_value=10,
        value=25,
        step=5,
        key="demo_min_points_part",
    )
    demo_prefer_surface_module = st.checkbox(
        "Предпочесть SurfaceReconstructor в демо",
        value=True,
        key="demo_prefer_surface_module",
    )
    st.caption("Демо-сценарий работает без весов .ckpt.")
    demo_clear_prev = st.checkbox(
        "Очистить предыдущий демо-результат перед запуском",
        value=False,
        key="demo_clear_prev",
    )

    if st.button("Запустить демо-сценарий 'идеальная работа'"):
        try:
            demo_path_clean = _resolve_file_input(
                path_value=demo_cloud_path,
                uploaded_file=uploaded_demo_cloud,
                output_root=output_root,
                prefix="demo_cloud",
                session_key="demo_cloud_path",
                fallback_suffix=".ply",
            )
            if not demo_path_clean:
                raise ValueError("Укажите путь к заранее размеченному облаку.")
            points, labels = load_point_cloud_file(demo_path_clean)
            object_class_name = demo_object_class.strip() or "valve"
            object_id = f"{object_class_name}_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            demo_output_dir = Path(output_root) / "demo_mode" / object_id

            if demo_clear_prev and demo_output_dir.exists():
                for child in demo_output_dir.iterdir():
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)

            with st.spinner("Запускаем демо: распознавание -> сегментация -> восстановление..."):
                classification = build_classification_from_labeled_cloud(
                    points=points,
                    labels=labels,
                    source="prelabeled_demo",
                )
                if len(np.unique(classification.labels)) <= 1:
                    demo_virtual_object = build_virtual_object(
                        points=points,
                        instance_id=object_id,
                        object_class=object_class_name,
                        source_file=demo_path_clean,
                    )
                    classification = classify_object_parts(
                        obj=demo_virtual_object,
                        checkpoint_path=None,
                        num_points=4096,
                    )
                    classification.notes.insert(
                        0,
                        "Демо: метки сегментации в исходном облаке не найдены, применена автоматическая геометрическая сегментация без .ckpt.",
                    )
                reconstruction = reconstruct_object_surfaces(
                    object_id=object_id,
                    classification=classification,
                    output_dir=demo_output_dir,
                    min_points_per_part=int(demo_min_points_part),
                    prefer_surface_module=bool(demo_prefer_surface_module),
                    checkpoint_path=None,
                )
                obj_export = export_reconstruction_obj(
                    reconstruction,
                    output_path=demo_output_dir / f"{object_id}_reconstruction.obj",
                )
        except Exception as demo_error:
            st.session_state.pop("demo_mode_state", None)
            st.error(f"Ошибка демо-сценария: {demo_error}")
        else:
            unique_labels, counts = np.unique(classification.labels, return_counts=True)
            st.session_state["demo_mode_state"] = {
                "object_id": object_id,
                "object_class": object_class_name,
                "source_path": demo_path_clean,
                "input_points": points.astype(np.float32, copy=False),
                "input_labels": labels.astype(np.int32, copy=False),
                "classification": classification,
                "reconstruction": reconstruction,
                "obj_export": obj_export,
                "histogram": {int(label): int(count) for label, count in zip(unique_labels, counts)},
            }
            st.success("Демо-сценарий выполнен.")

    demo_mode_state = st.session_state.get("demo_mode_state")
    if demo_mode_state is not None:
        st.markdown("### Демонстрация шагов")
        st.write(
            f"Объект: **{demo_mode_state['object_id']}** | "
            f"Тип: **{demo_mode_state['object_class']}** | "
            f"Источник: `{demo_mode_state['source_path']}`"
        )

        st.markdown("#### Шаг 1. Распознавание объекта (визуально без разметки)")
        st.success(
            f"Объект распознан как `{demo_mode_state['object_class']}`. "
            "Ниже показывается входное облако без сегментных меток."
        )
        st.plotly_chart(
            _build_cloud_figure(
                demo_mode_state["input_points"],
                None,
                title="Демо: распознанный объект (без разметки)",
                max_points=90000,
                point_size=2,
            ),
            use_container_width=True,
        )

        st.markdown("#### Шаг 2. Сегментация")
        st.plotly_chart(
            _build_cloud_figure(
                demo_mode_state["classification"].points,
                demo_mode_state["classification"].labels,
                title="Демо: сегментация завершена",
                max_points=90000,
                point_size=3,
            ),
            use_container_width=True,
        )
        st.dataframe(
            _class_hist_table(demo_mode_state["histogram"]),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### Шаг 3. Восстановление и демонстрация OBJ")
        reconstruction = demo_mode_state["reconstruction"]
        obj_export = demo_mode_state["obj_export"]
        st.write(
            f"Результат восстановления: backend=`{reconstruction.backend}`, "
            f"файлов=`{len(reconstruction.generated_files)}`"
        )
        if reconstruction.notes:
            for note in reconstruction.notes:
                st.warning(note)

        st.write(f"Восстановленный OBJ: `{obj_export['obj_path']}` (режим: `{obj_export['mode']}`)")
        if obj_export.get("mesh_error"):
            st.caption(f"Примечание экспорта OBJ: {obj_export['mesh_error']}")

        try:
            obj_fig, obj_vertices, obj_faces = _build_obj_figure(
                obj_export["obj_path"],
                title="Демо: восстановленный OBJ",
            )
            st.plotly_chart(obj_fig, use_container_width=True)
            st.caption(f"OBJ вершин: {obj_vertices} | треугольников (предпросмотр): {obj_faces}")
        except Exception as obj_preview_error:
            st.warning(f"Не удалось построить предпросмотр OBJ: {obj_preview_error}")

        if len(reconstruction.combined_points) > 0:
            st.plotly_chart(
                _build_cloud_figure(
                    reconstruction.combined_points,
                    None,
                    title="Восстановленная геометрия (облако точек)",
                    max_points=90000,
                    point_size=2,
                ),
                use_container_width=True,
            )
else:
    st.session_state.pop("demo_mode_state", None)

st.markdown("## 1) Загрузка готового облака для анализа")
analysis_cloud_path = st.text_input("Путь к облаку точки (.ply)", value="", key="analysis_cloud_path_input")
uploaded_analysis_cloud = st.file_uploader(
    "Или выберите файл облака через диалог",
    type=["ply"],
    key="analysis_cloud_upload",
)
if st.button("Загрузить облако для анализа"):
    try:
        analysis_cloud_path_clean = _resolve_file_input(
            path_value=analysis_cloud_path,
            uploaded_file=uploaded_analysis_cloud,
            output_root=output_root,
            prefix="analysis_cloud",
            session_key="analysis_cloud_path_input",
            fallback_suffix=".ply",
        )
        if not analysis_cloud_path_clean:
            raise ValueError("Укажите путь к .ply или выберите файл через диалог.")

        points, labels = load_point_cloud_file(analysis_cloud_path_clean)
    except Exception as load_error:
        st.error(f"Ошибка загрузки облака: {load_error}")
    else:
        st.session_state["analysis_cloud_state"] = {
            "path": analysis_cloud_path_clean,
            "points": points,
            "labels": labels,
        }
        st.success("Облако загружено.")

analysis_cloud_state = st.session_state.get("analysis_cloud_state")
if analysis_cloud_state is not None:
    c1, c2 = st.columns(2)
    c1.metric("Точек в облаке", int(len(analysis_cloud_state["points"])))
    c2.metric(
        "Диапазон меток",
        f"{int(analysis_cloud_state['labels'].min())}..{int(analysis_cloud_state['labels'].max())}",
    )
    st.write(f"Источник: `{analysis_cloud_state['path']}`")
    st.plotly_chart(
        _build_cloud_figure(
            analysis_cloud_state["points"],
            analysis_cloud_state["labels"],
            title="Загруженное облако",
            max_points=90000,
            point_size=2,
        ),
        use_container_width=True,
    )
else:
    st.info("Загрузите облако, чтобы запустить распознавание.")

st.markdown("## 2) Распознавание объектов")
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
    key="recognition_ckpt_input",
)
uploaded_recognition_ckpt = st.file_uploader(
    "Или выберите checkpoint распознавания",
    type=["ckpt"],
    key="recognition_ckpt_upload",
)
recognition_points = st.number_input("Точек на инференс", min_value=256, value=4096, step=256)

if st.button("Запустить распознавание объектов в облаке"):
    cloud = st.session_state.get("analysis_cloud_state")
    recognition_ckpt_path = _resolve_file_input(
        path_value=recognition_ckpt,
        uploaded_file=uploaded_recognition_ckpt,
        output_root=output_root,
        prefix="recognition_ckpt",
        session_key="recognition_ckpt_input",
        fallback_suffix=".ckpt",
    ) or auto_recognition_ckpt
    if cloud is None:
        st.error("Сначала загрузите облако для анализа.")
    elif not recognition_ckpt_path:
        st.error("Укажите путь к .ckpt модели распознавания или обучите её в блоке ниже.")
    else:
        with st.spinner("Распознавание выполняется..."):
            recognition = recognize_cloud_with_model(
                points=cloud["points"],
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
                title="Распознанные метки",
                max_points=80000,
                point_size=2,
            ),
            use_container_width=True,
        )

st.markdown("### Дообучение модели распознавания")
st.caption("Формат датасета распознавания: каждая папка = отдельный тип объекта, внутри папки — варианты поворотов/поз.")
rec_t1, rec_t2, rec_t3 = st.columns(3)
with rec_t1:
    train_rec_dataset_root = st.text_input(
        "Папка с наборами объектов (каждая папка = один тип)",
        value=str(DEFAULT_DATASET_DIR),
        key="analysis_train_rec_dataset_root",
    )
    train_rec_model_name = st.text_input(
        "Имя модели распознавания (папка весов)",
        value="factory_recognition",
        key="analysis_train_rec_model_name",
    )
with rec_t2:
    train_rec_num_classes = st.number_input(
        "Классов",
        min_value=2,
        value=13,
        step=1,
        key="analysis_train_rec_num_classes",
    )
    train_rec_epochs = st.number_input(
        "Эпох",
        min_value=1,
        value=20,
        step=1,
        key="analysis_train_rec_epochs",
    )
with rec_t3:
    train_rec_batch = st.number_input(
        "Batch size",
        min_value=1,
        value=4,
        step=1,
        key="analysis_train_rec_batch",
    )
    train_rec_lr = st.number_input(
        "Learning rate",
        min_value=0.00001,
        value=0.001,
        step=0.0001,
        format="%.5f",
        key="analysis_train_rec_lr",
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
    label_space = recognition_finetune_state.get("label_space") or {}
    if recognition_finetune_state.get("label_mode") == "folder_name" and label_space:
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
    st.write(
        {
            "model_name": recognition_finetune_state.get("object_name"),
            "weights_root": recognition_finetune_state.get("weights_root"),
            "best_checkpoint": recognition_finetune_state.get("best_checkpoint"),
            "test_metrics": recognition_finetune_state.get("test_metrics"),
        }
    )

st.markdown("## 3) Сегментация выбранного объекта")
st.caption("Вы можете сегментировать объект из отдельного облака `.ply`.")

st.markdown("### Дообучение модели сегментации")
seg_t1, seg_t2, seg_t3 = st.columns(3)
with seg_t1:
    train_seg_dataset_root = st.text_input(
        "Папка с облаками точек объекта для сегментации",
        value=str(DEFAULT_DATASET_DIR),
        key="analysis_train_seg_dataset_root",
    )
    train_seg_object_name = st.text_input(
        "Имя типа объекта (папка весов сегментации)",
        value="valve",
        key="analysis_train_seg_object_name",
    )
with seg_t2:
    train_seg_num_classes = st.number_input(
        "Классов",
        min_value=2,
        value=13,
        step=1,
        key="analysis_train_seg_num_classes",
    )
    train_seg_epochs = st.number_input(
        "Эпох",
        min_value=1,
        value=20,
        step=1,
        key="analysis_train_seg_epochs",
    )
with seg_t3:
    train_seg_batch = st.number_input(
        "Batch size",
        min_value=1,
        value=4,
        step=1,
        key="analysis_train_seg_batch",
    )
    train_seg_lr = st.number_input(
        "Learning rate",
        min_value=0.00001,
        value=0.001,
        step=0.0001,
        format="%.5f",
        key="analysis_train_seg_lr",
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
    st.write(
        {
            "object_name": segmentation_finetune_state.get("object_name"),
            "weights_root": segmentation_finetune_state.get("weights_root"),
            "best_checkpoint": segmentation_finetune_state.get("best_checkpoint"),
            "test_metrics": segmentation_finetune_state.get("test_metrics"),
        }
    )

obj_col1, obj_col2 = st.columns(2)
with obj_col1:
    seg_object_cloud_path = st.text_input(
        "Путь к облаку объекта для сегментации (.ply)",
        value="",
        key="seg_object_cloud_path_input",
    )
    uploaded_seg_object_cloud = st.file_uploader(
        "Или выберите облако объекта для сегментации",
        type=["ply"],
        key="seg_object_cloud_upload",
    )
with obj_col2:
    seg_object_class = st.text_input("Тип объекта", value="valve")

auto_seg_ckpt = _find_auto_weights(Path(segmentation_weights_root), seg_object_class.strip() or "valve")
if auto_seg_ckpt:
    st.caption(f"Автоматически найден вес сегментации: `{auto_seg_ckpt}`")
seg_ckpt_input = st.text_input(
    "Путь к весам сегментации (.ckpt, можно оставить пустым для авто)",
    value="",
    key="seg_ckpt_input_path",
)
uploaded_seg_ckpt = st.file_uploader(
    "Или выберите checkpoint сегментации",
    type=["ckpt"],
    key="seg_ckpt_upload",
)

if st.button("Запустить сегментацию объекта"):
    ckpt_path = _resolve_file_input(
        path_value=seg_ckpt_input,
        uploaded_file=uploaded_seg_ckpt,
        output_root=output_root,
        prefix="segmentation_ckpt",
        session_key="seg_ckpt_input_path",
        fallback_suffix=".ckpt",
    ) or auto_seg_ckpt
    seg_object_cloud_path_clean = _resolve_file_input(
        path_value=seg_object_cloud_path,
        uploaded_file=uploaded_seg_object_cloud,
        output_root=output_root,
        prefix="seg_object_cloud",
        session_key="seg_object_cloud_path_input",
        fallback_suffix=".ply",
    )
    if not seg_object_cloud_path_clean:
        st.error("Укажите путь к облаку объекта.")
    else:
        try:
            points, _ = load_point_cloud_file(seg_object_cloud_path_clean)
            object_id = f"{(seg_object_class.strip() or 'object')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            virtual_object = build_virtual_object(
                points=points,
                instance_id=object_id,
                object_class=seg_object_class.strip() or "object",
                source_file=seg_object_cloud_path_clean,
            )
            with st.spinner("Сегментация выполняется..."):
                segmentation = classify_object_parts(
                    obj=virtual_object,
                    checkpoint_path=ckpt_path or None,
                    num_points=4096,
                )
        except Exception as seg_error:
            st.session_state.pop("segmentation_state", None)
            st.error(f"Ошибка сегментации: {seg_error}")
        else:
            st.session_state["segmentation_state"] = {
                "object_id": object_id,
                "object_class": virtual_object.object_class,
                "source_file": seg_object_cloud_path_clean,
                "classification": segmentation,
            }
            st.success("Сегментация завершена.")

segmentation_state = st.session_state.get("segmentation_state")
if segmentation_state is not None:
    classification = segmentation_state["classification"]
    if classification.notes:
        for note in classification.notes:
            st.warning(note)
    s1, s2 = st.columns([1, 2])
    with s1:
        st.write(f"Объект: **{segmentation_state['object_id']}**")
        st.write(f"Источник: `{segmentation_state['source_file']}`")
        st.dataframe(_class_hist_table(classification.histogram), hide_index=True, use_container_width=True)
    with s2:
        st.plotly_chart(
            _build_cloud_figure(
                classification.points,
                classification.labels,
                title=f"Сегментация объекта {segmentation_state['object_id']}",
                max_points=70000,
                point_size=3,
            ),
            use_container_width=True,
        )

st.markdown("## 4) Восстановление поверхности")
if segmentation_state is None:
    st.info("Сначала выполните сегментацию объекта.")
else:
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        min_points_part = st.number_input("Минимум точек на часть", min_value=10, value=45, step=5)
    with rc2:
        prefer_surface_module = st.checkbox("Предпочесть SurfaceReconstructor.py", value=False)
    with rc3:
        surface_ckpt = st.text_input(
            "Checkpoint для SurfaceReconstructor (опционально)",
            value="",
            key="surface_ckpt_input",
        )
        uploaded_surface_ckpt = st.file_uploader(
            "Или выберите checkpoint SurfaceReconstructor",
            type=["ckpt"],
            key="surface_ckpt_upload",
        )

    if st.button("Запустить восстановление поверхности"):
        surface_ckpt_path = _resolve_file_input(
            path_value=surface_ckpt,
            uploaded_file=uploaded_surface_ckpt,
            output_root=output_root,
            prefix="surface_reconstructor_ckpt",
            session_key="surface_ckpt_input",
            fallback_suffix=".ckpt",
        )
        with st.spinner("Восстановление поверхности выполняется..."):
            reconstruction = reconstruct_object_surfaces(
                object_id=segmentation_state["object_id"],
                classification=segmentation_state["classification"],
                output_dir=Path(output_root) / "reconstruction" / segmentation_state["object_id"],
                min_points_per_part=int(min_points_part),
                prefer_surface_module=bool(prefer_surface_module),
                checkpoint_path=surface_ckpt_path or None,
            )
        st.session_state["reconstruction_state"] = reconstruction
        st.success("Восстановление завершено.")

reconstruction_state = st.session_state.get("reconstruction_state")
if reconstruction_state is not None:
    st.write(f"Backend: **{reconstruction_state.backend}**")
    st.write(f"Сгенерировано файлов: **{len(reconstruction_state.generated_files)}**")
    if reconstruction_state.notes:
        for note in reconstruction_state.notes:
            st.warning(note)
    if reconstruction_state.generated_files:
        st.write("Файлы:")
        for file_path in reconstruction_state.generated_files:
            st.write(f"- `{file_path}`")
    if len(reconstruction_state.combined_points) > 0:
        st.plotly_chart(
            _build_cloud_figure(
                reconstruction_state.combined_points,
                None,
                title="Восстановленная поверхность",
                max_points=90000,
                point_size=2,
            ),
            use_container_width=True,
        )
