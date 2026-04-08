from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ПО для инженерных объектов",
    page_icon="",
    layout="wide",
)

st.title("Программное обеспечение для инженерных объектов")
st.caption("Единая программа с двумя вкладками: синтетическое создание данных и анализ готовых облаков.")

tab_mode = st.radio(
    "Вкладка",
    options=[
        "Конструктор синтетических облаков",
        "Анализ готовых облаков",
    ],
    horizontal=True,
)

if tab_mode == "Конструктор синтетических облаков":
    runpy.run_path(
        str(BASE_DIR / "factory_synthetic_builder_app.py"),
        init_globals={"EMBEDDED_MODE": True},
    )
else:
    runpy.run_path(
        str(BASE_DIR / "factory_analysis_app.py"),
        init_globals={"EMBEDDED_MODE": True},
    )
