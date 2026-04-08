from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataset import sanitize_filesystem_path
from engineering_classifier import find_latest_checkpoint
from factory_showcase_pipeline import (
    build_virtual_object,
    classify_object_parts,
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


def _class_hist_table(hist: dict[int, int]) -> pd.DataFrame:
    rows = [{"Метка": int(label), "Точек": int(count)} for label, count in sorted(hist.items())]
    if not rows:
        return pd.DataFrame(columns=["Метка", "Точек"])
    return pd.DataFrame(rows)


def _find_auto_weights(weights_root: Path, object_class: str) -> str:
    object_folder = weights_root / object_class / "checkpoints"
    latest = find_latest_checkpoint([object_folder])
    return latest or ""


def _ensure_state() -> None:
    st.session_state.setdefault("analysis_cloud_state", None)
    st.session_state.setdefault("recognition_state", None)
    st.session_state.setdefault("recognition_finetune_state", None)
    st.session_state.setdefault("segmentation_finetune_state", None)
    st.session_state.setdefault("segmentation_state", None)
    st.session_state.setdefault("reconstruction_state", None)


if not EMBEDDED_MODE:
    st.set_page_config(
        page_title="Анализ готовых облаков",
        page_icon="",
        layout="wide",
    )

_ensure_state()

st.title("Аналитика готовых облаков точек")
st.caption(
    "Отдельная часть для работы с уже подготовленными облаками: распознавание, сегментация и восстановление поверхностей."
)

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

st.markdown("## 1) Загрузка готового облака для анализа")
analysis_cloud_path = st.text_input("Путь к облаку точки (.ply)", value="")
if st.button("Загрузить облако для анализа"):
    try:
        analysis_cloud_path_clean = sanitize_filesystem_path(analysis_cloud_path)
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
)
recognition_points = st.number_input("Точек на инференс", min_value=256, value=4096, step=256)

if st.button("Запустить распознавание объектов в облаке"):
    cloud = st.session_state.get("analysis_cloud_state")
    recognition_ckpt_path = recognition_ckpt.strip() or auto_recognition_ckpt
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
    seg_object_cloud_path = st.text_input("Путь к облаку объекта для сегментации (.ply)", value="")
with obj_col2:
    seg_object_class = st.text_input("Тип объекта", value="valve")

auto_seg_ckpt = _find_auto_weights(Path(segmentation_weights_root), seg_object_class.strip() or "valve")
if auto_seg_ckpt:
    st.caption(f"Автоматически найден вес сегментации: `{auto_seg_ckpt}`")
seg_ckpt_input = st.text_input(
    "Путь к весам сегментации (.ckpt, можно оставить пустым для авто)",
    value="",
)

if st.button("Запустить сегментацию объекта"):
    ckpt_path = seg_ckpt_input.strip() or auto_seg_ckpt
    seg_object_cloud_path_clean = sanitize_filesystem_path(seg_object_cloud_path)
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
        prefer_surface_module = st.checkbox("Предпочесть SurfaceReconstructor.py", value=True)
    with rc3:
        surface_ckpt = st.text_input("Checkpoint для SurfaceReconstructor (опционально)", value="")

    if st.button("Запустить восстановление поверхности"):
        with st.spinner("Восстановление поверхности выполняется..."):
            reconstruction = reconstruct_object_surfaces(
                object_id=segmentation_state["object_id"],
                classification=segmentation_state["classification"],
                output_dir=Path(output_root) / "reconstruction" / segmentation_state["object_id"],
                min_points_per_part=int(min_points_part),
                prefer_surface_module=bool(prefer_surface_module),
                checkpoint_path=surface_ckpt.strip() or None,
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
