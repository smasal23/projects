import streamlit as st
from app.utils.data_loader import safe_get

def render_stakeholders(data):

    st.title("👥 Stakeholder Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎯 Beneficiaries")
        for item in safe_get(data, "stakeholder_report.beneficiaries", []):
            st.markdown(f"<div class='card'>{item}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("⚠️ Constraints")
        for item in safe_get(data, "stakeholder_report.constraints", []):
            st.markdown(f"<div class='card warning'>{item}</div>", unsafe_allow_html=True)

    with col3:
        st.subheader("🚀 Opportunities")
        for item in safe_get(data, "stakeholder_report.opportunities", []):
            st.markdown(f"<div class='card success'>{item}</div>", unsafe_allow_html=True)

    # -------------------------
    # Visualization
    # -------------------------
    st.subheader("📊 Stakeholder Distribution")

    counts = {
        "Beneficiaries": len(safe_get(data, "stakeholder_report.beneficiaries", [])),
        "Constraints": len(safe_get(data, "stakeholder_report.constraints", [])),
        "Opportunities": len(safe_get(data, "stakeholder_report.opportunities", []))
    }

    import pandas as pd
    df = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])

    st.bar_chart(df.set_index("Category"))