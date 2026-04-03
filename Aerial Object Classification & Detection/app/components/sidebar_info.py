from __future__ import annotations

import streamlit as st

from src.app.ui_helpers import render_sidebar_project_info


def show_sidebar_info(streamlit_config: dict) -> None:
    """
    Render project info in sidebar.
    """
    render_sidebar_project_info(streamlit_config)
    st.sidebar.markdown("---")
    st.sidebar.caption("Built for aerial image classification and detection workflows.")