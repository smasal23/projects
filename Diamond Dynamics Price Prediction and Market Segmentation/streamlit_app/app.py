import yaml
print("yaml installed successfully")

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_project_configs
from src.utils.paths import find_project_root

from streamlit_app.utils.constants import (
    DEFAULT_PAGE_ICON,
    DEFAULT_LAYOUT,
)
from streamlit_app.components.charts import (
    render_sidebar_project_info,
    render_home_feature_cards,
    render_quick_stats,
    render_how_it_works,
)


def load_app_context():
    root = find_project_root(PROJECT_ROOT)
    configs = load_project_configs(root)
    return root, configs


def load_css(project_root: Path):
    css_path = project_root / "streamlit_app" / "assets" / "custom_styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


project_root, configs = load_app_context()
streamlit_cfg = configs["streamlit_config"]
main_cfg = configs["main_config"]
features_cfg = configs["features_config"]

st.set_page_config(
    page_title=streamlit_cfg["streamlit"].get("app_title", "Diamond App"),
    page_icon=streamlit_cfg["streamlit"].get("page_icon", DEFAULT_PAGE_ICON),
    layout=streamlit_cfg["streamlit"].get("layout", DEFAULT_LAYOUT),
    initial_sidebar_state=streamlit_cfg["streamlit"].get("initial_sidebar_state", "expanded"),
)

st.set_option("client.showSidebarNavigation", False)

load_css(project_root)

app_title = streamlit_cfg["streamlit"].get("app_title", "Diamond Price Prediction & Market Segmentation")
header_title = streamlit_cfg["branding"].get("header_title", "Diamond Price Predictor & Segment Explorer")
header_subtitle = streamlit_cfg["branding"].get(
    "header_subtitle",
    "Estimate diamond price and identify market segment from raw gem attributes.",
)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-badge">Diamond Dynamics ML App</div>
        <h1>{header_title}</h1>
        <p>{header_subtitle}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

banner_path = project_root / "streamlit_app" / "assets" / "diamond_banner.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)

render_sidebar_project_info(configs)
render_quick_stats(features_cfg, main_cfg)
render_home_feature_cards()
render_how_it_works()

st.markdown("---")
st.subheader("Use the app")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>💰 Price Prediction</h3>
            <p>Enter diamond attributes and estimate the predicted price using the saved regression pipeline.</p>
            <p><strong>Output:</strong> Predicted price in INR and derived price band.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>📊 Market Segmentation</h3>
            <p>Use the same input attributes to assign the diamond to a learned customer/product segment.</p>
            <p><strong>Output:</strong> Cluster number and mapped cluster label.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.info(
    "Open the sidebar pages to start predictions. Both modules use the same validated input schema."
)

if streamlit_cfg["branding"].get("show_footer", True):
    st.markdown("---")
    st.caption(streamlit_cfg["branding"].get("footer_text", "Built with Streamlit for Diamond Dynamics"))
