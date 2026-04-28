import streamlit as st
from app.utils.data_loader import safe_get

def render_summary(data):

    st.title("📌 Regulation Topic")
    st.markdown(f"### {safe_get(data, 'regulation_topic', 'N/A')}")

    st.title("📘 Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎓 Students")
        for item in safe_get(data, "summary.student_view", []):
            st.markdown(f"<div class='card'>{item}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("👨‍🏫 Faculty")
        for item in safe_get(data, "summary.faculty_view", []):
            st.markdown(f"<div class='card'>{item}</div>", unsafe_allow_html=True)

    with col3:
        st.subheader("🏫 Institutions")
        for item in safe_get(data, "summary.institution_view", []):
            st.markdown(f"<div class='card'>{item}</div>", unsafe_allow_html=True)