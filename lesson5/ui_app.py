from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from dataset import load_ply_file
from engineering_classifier import (
    TrainingSettings,
    build_point_cloud_figure,
    collect_dataset_statistics,
    find_latest_checkpoint,
    list_ply_files,
    predict_file,
    save_predictions_to_ply,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "3011"


def _parse_class_weights(raw_value: str) -> list[float] | None:
    value = raw_value.strip()
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _format_distribution_table(distribution: dict[int, int]) -> pd.DataFrame:
    if not distribution:
        return pd.DataFrame(columns=["class_id", "count", "share_percent"])
    total = sum(distribution.values())
    rows = []
    for class_id, count in distribution.items():
        share = (count / total) * 100.0 if total > 0 else 0.0
        rows.append({"class_id": class_id, "count": count, "share_percent": round(share, 2)})
    return pd.DataFrame(rows).sort_values("class_id")


def _determine_default_checkpoint() -> str:
    candidate = st.session_state.get("last_checkpoint")
    if candidate:
        return str(candidate)

    latest = find_latest_checkpoint(
        [BASE_DIR / "checkpoints_ui", BASE_DIR / "checkpoints"]
    )
    return latest or ""


st.set_page_config(
    page_title="Point Cloud Engineering Classifier",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: "Manrope", sans-serif;
}

.stApp {
    background: radial-gradient(circle at 0% 0%, #e6efff 0%, #f5f8ff 46%, #fdfdff 100%);
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

.mono-label {
    font-family: "IBM Plex Mono", monospace;
    color: #2d4674;
}
</style>
    """,
    unsafe_allow_html=True,
)

st.title("Engineering Object Classifier for Point Clouds")
st.caption(
    "Neural segmentation of engineering objects with a single control interface: data, training, and inference."
)

with st.sidebar:
    st.subheader("Control Panel")
    dataset_root = st.text_input(
        "Dataset folder (.ply files)",
        value=str(DEFAULT_DATASET_DIR),
    )
    max_scan_files = st.slider(
        "Files to scan for statistics",
        min_value=10,
        max_value=300,
        value=80,
        step=10,
    )
    st.markdown(
        f"<span class='mono-label'>Compute device: {'CUDA' if torch.cuda.is_available() else 'CPU/MPS'}</span>",
        unsafe_allow_html=True,
    )

tab_data, tab_train, tab_infer = st.tabs(["Data", "Training", "Inference"])

with tab_data:
    st.subheader("Dataset Overview")
    data_files = list_ply_files(dataset_root)
    if not data_files:
        st.warning(f"No .ply files found in `{dataset_root}`.")
    else:
        stats_key = "dataset_stats"
        root_key = "dataset_stats_root"
        if (
            stats_key not in st.session_state
            or st.session_state.get(root_key) != dataset_root
        ):
            with st.spinner("Collecting dataset statistics..."):
                st.session_state[stats_key] = collect_dataset_statistics(
                    dataset_root, max_files=max_scan_files
                )
                st.session_state[root_key] = dataset_root

        if st.button("Refresh statistics", key="refresh_dataset_stats"):
            with st.spinner("Re-scanning dataset..."):
                st.session_state[stats_key] = collect_dataset_statistics(
                    dataset_root, max_files=max_scan_files
                )
                st.session_state[root_key] = dataset_root

        stats = st.session_state[stats_key]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total files", stats["file_count"])
        col2.metric("Scanned files", stats["scanned_files"])
        col3.metric("Min points/file", stats["point_count_min"])
        col4.metric("Mean points/file", f"{stats['point_count_mean']:.1f}")

        st.markdown("Class distribution (from scanned subset):")
        st.dataframe(
            _format_distribution_table(stats["class_distribution"]),
            use_container_width=True,
            hide_index=True,
        )

        preview_candidates = [str(path) for path in data_files[:300]]
        selected_preview = st.selectbox("Preview point cloud file", options=preview_candidates)
        max_render_points = st.slider(
            "Max points for preview rendering",
            min_value=2000,
            max_value=60000,
            value=25000,
            step=1000,
        )

        preview_points, preview_labels = load_ply_file(selected_preview)
        if preview_labels is None or len(preview_labels) == 0:
            preview_labels = None

        preview_figure = build_point_cloud_figure(
            preview_points,
            preview_labels,
            title=f"Preview: {Path(selected_preview).name}",
            max_points=max_render_points,
        )
        st.plotly_chart(preview_figure, use_container_width=True)

with tab_train:
    st.subheader("Training Workflow")
    st.write("Configure and run model training directly from this interface.")

    with st.form("training_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            train_num_classes = st.number_input("Classes", min_value=2, value=13, step=1)
            train_num_points = st.number_input("Points per sample", min_value=128, value=4096, step=128)
            train_batch_size = st.number_input("Batch size", min_value=1, value=4, step=1)
            train_epochs = st.number_input("Epochs", min_value=1, value=60, step=1)
        with col_b:
            train_lr = st.number_input("Learning rate", min_value=0.00001, value=0.001, step=0.0001, format="%.5f")
            train_weight_decay = st.number_input(
                "Weight decay", min_value=0.0, value=0.0001, step=0.0001, format="%.5f"
            )
            train_scheduler_step = st.number_input("Scheduler step", min_value=1, value=20, step=1)
            train_scheduler_gamma = st.number_input(
                "Scheduler gamma", min_value=0.1, max_value=1.0, value=0.7, step=0.05, format="%.2f"
            )
        with col_c:
            train_val_size = st.number_input(
                "Validation split", min_value=0.05, max_value=0.45, value=0.1, step=0.05, format="%.2f"
            )
            train_test_size = st.number_input(
                "Test split", min_value=0.05, max_value=0.45, value=0.1, step=0.05, format="%.2f"
            )
            train_seed = st.number_input("Random seed", min_value=1, value=42, step=1)
            train_patience = st.number_input("Early stopping patience", min_value=1, value=15, step=1)

        class_weights_input = st.text_input(
            "Class weights (optional, comma separated)",
            placeholder="Example: 0.2, 1.0, 0.8, 1.3",
        )

        training_submitted = st.form_submit_button("Start training")

    if training_submitted:
        if train_val_size + train_test_size >= 0.95:
            st.error("Validation and test splits are too large. Keep at least 5% for training.")
        else:
            try:
                class_weights = _parse_class_weights(class_weights_input)
            except ValueError:
                st.error("Class weights must be numeric values separated by commas.")
                class_weights = None

            if class_weights_input.strip() and class_weights is None:
                st.stop()

            settings = TrainingSettings(
                dataset_root=dataset_root,
                num_classes=int(train_num_classes),
                num_points=int(train_num_points),
                val_size=float(train_val_size),
                test_size=float(train_test_size),
                batch_size=int(train_batch_size),
                num_workers=0,
                max_epochs=int(train_epochs),
                learning_rate=float(train_lr),
                weight_decay=float(train_weight_decay),
                scheduler_step_size=int(train_scheduler_step),
                scheduler_gamma=float(train_scheduler_gamma),
                early_stopping_patience=int(train_patience),
                class_weights=class_weights,
                seed=int(train_seed),
                checkpoint_dir=str(BASE_DIR / "checkpoints_ui"),
                log_dir=str(BASE_DIR / "logs_ui"),
            )

            logs_placeholder = st.empty()
            status_messages: list[str] = []

            def _status(message: str) -> None:
                status_messages.append(message)
                logs_placeholder.code("\n".join(status_messages[-12:]))

            try:
                with st.spinner("Training in progress. This can take time..."):
                    training_result = train_model(settings, status_cb=_status)
                st.session_state["last_training_result"] = training_result
                st.session_state["last_checkpoint"] = training_result["best_checkpoint"]
                st.success("Training and evaluation are completed.")
            except Exception as error:
                st.exception(error)

    if "last_training_result" in st.session_state:
        st.markdown("Latest training result:")
        latest_result = st.session_state["last_training_result"]
        st.json(
            {
                "best_checkpoint": latest_result.get("best_checkpoint"),
                "split_counts": latest_result.get("split_counts"),
                "test_metrics": latest_result.get("test_metrics"),
            }
        )

with tab_infer:
    st.subheader("Inference and Export")
    default_checkpoint = _determine_default_checkpoint()
    checkpoint_path = st.text_input(
        "Checkpoint path (.ckpt)",
        value=default_checkpoint,
    )

    source_mode = st.radio(
        "Input mode",
        options=["File path", "Upload file"],
        horizontal=True,
    )
    selected_input_path = ""
    if source_mode == "File path":
        selected_input_path = st.text_input(
            "Point cloud file path (.ply)",
            value=str(data_files[0]) if data_files else "",
        )
    else:
        uploaded_file = st.file_uploader("Upload a .ply file", type=["ply"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                selected_input_path = temp_file.name

    infer_num_points = st.number_input(
        "Points sampled for inference",
        min_value=128,
        value=4096,
        step=128,
    )

    if st.button("Run inference", key="run_inference_button"):
        try:
            inference_result = predict_file(
                checkpoint_path=checkpoint_path,
                input_file=selected_input_path,
                num_points=int(infer_num_points),
            )
            st.session_state["last_inference_result"] = inference_result
            st.success("Inference completed.")
        except Exception as error:
            st.exception(error)

    if "last_inference_result" in st.session_state:
        result = st.session_state["last_inference_result"]
        st.metric("Mean confidence", f"{result['mean_confidence']:.3f}")

        histogram_df = pd.DataFrame(result["prediction_histogram"])
        if not histogram_df.empty:
            st.dataframe(histogram_df, hide_index=True, use_container_width=True)

        if result.get("has_true_labels"):
            col_pred, col_true = st.columns(2)
            with col_pred:
                st.plotly_chart(
                    build_point_cloud_figure(
                        result["points"],
                        result["predicted_labels"],
                        title="Predicted labels",
                    ),
                    use_container_width=True,
                )
            with col_true:
                st.plotly_chart(
                    build_point_cloud_figure(
                        result["points"],
                        result["true_labels"],
                        title="True labels",
                    ),
                    use_container_width=True,
                )
        else:
            st.plotly_chart(
                build_point_cloud_figure(
                    result["points"],
                    result["predicted_labels"],
                    title="Predicted labels",
                ),
                use_container_width=True,
            )

        default_save_path = BASE_DIR / "predictions" / f"pred_{Path(result['input_file']).name}"
        save_path = st.text_input(
            "Save predictions to",
            value=str(default_save_path),
            key="save_prediction_path",
        )

        if st.button("Save predicted point cloud", key="save_predicted_cloud"):
            try:
                saved = save_predictions_to_ply(
                    output_path=save_path,
                    points=result["points"],
                    predicted_labels=result["predicted_labels"],
                )
                st.success(f"Saved to: {saved}")
            except Exception as error:
                st.exception(error)
