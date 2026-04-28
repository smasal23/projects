import streamlit as st
from app.utils.data_loader import safe_get

def render_impact(data):

    st.title("📈 Impact Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚡ Short Term")
        for item in safe_get(data, "impact_assessment.short_term", []):
            st.markdown(f"<div class='card'>{item}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Medium Term")
        for item in safe_get(data, "impact_assessment.medium_term", []):
            st.markdown(f"<div class='card'>{item}</div>", unsafe_allow_html=True)

    with col3:
        st.subheader("🏁 Long Term")
        for item in safe_get(data, "impact_assessment.long_term", []):
            st.markdown(f"<div class='card'>{item}</div>", unsafe_allow_html=True)