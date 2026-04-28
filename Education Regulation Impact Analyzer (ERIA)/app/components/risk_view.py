import streamlit as st
from app.utils.data_loader import safe_get

def render_risks(data):

    st.title("⚠️ Risk & Sentiment Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("✅ Positives")
        for item in safe_get(data, "positives", []):
            st.markdown(f"<div class='card success'>{item}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("❌ Negatives")
        for item in safe_get(data, "negatives", []):
            st.markdown(f"<div class='card danger'>{item}</div>", unsafe_allow_html=True)

    with col3:
        st.subheader("🚨 Risk Flags")
        for item in safe_get(data, "sentiment_risk.risk_flags", []):
            st.markdown(f"<div class='card warning'>{item}</div>", unsafe_allow_html=True)

    st.subheader("📊 Risk Distribution")

    counts = {
        "Positives": len(safe_get(data, "positives", [])),
        "Negatives": len(safe_get(data, "negatives", [])),
        "Risk Flags": len(safe_get(data, "sentiment_risk.risk_flags", []))
    }

    import pandas as pd
    df = pd.DataFrame(list(counts.items()), columns=["Type", "Count"])

    st.bar_chart(df.set_index("Type"))