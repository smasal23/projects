from __future__ import annotations

import streamlit as st


def show_image_uploader() -> object:
    """
    Render image uploader widget and return uploaded file object.
    """
    return st.file_uploader(
        "Upload an aerial image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP",
    )