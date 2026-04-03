from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.components.sidebar_info import show_sidebar_info
from src.utils.config import load_project_configs


CONFIGS = load_project_configs(PROJECT_ROOT / "configs")
STREAMLIT_CFG = CONFIGS["streamlit"]
APP_CFG = STREAMLIT_CFG["app"]

st.set_page_config(
    page_title=APP_CFG.get("title", "Aerial Object Classification & Detection"),
    page_icon=APP_CFG.get("page_icon", "🛰️"),
    layout=APP_CFG.get("layout", "wide"),
    initial_sidebar_state=APP_CFG.get("sidebar_state", "expanded"),
)

st.title(APP_CFG.get("title", "Aerial Object Classification & Detection"))
st.caption(STREAMLIT_CFG.get("content", {}).get("subtitle", ""))

show_sidebar_info(STREAMLIT_CFG)

st.markdown("## Welcome")
st.write(
    "Use the navigation in the sidebar to open the Classification or Detection page. "
    "This app is configured to load saved model artifacts directly from your project structure."
)

st.markdown("### Available Modes")
st.markdown("- **Classification:** Predict whether an aerial image contains a bird or a drone.")
if STREAMLIT_CFG.get("detection", {}).get("enabled", True):
    st.markdown("- **Detection:** Run YOLOv8 and visualize bounding boxes on the uploaded image.")

st.markdown("### Notes")
st.markdown("- Keep trained model artifacts in the configured `models/` directories.")
st.markdown("- Keep class mapping JSON files available for classifier inference.")
st.markdown("- Capture the full UI screenshots manually after launching the app.")