import streamlit as st
import pandas as pd
import graphviz
from app.utils.data_loader import safe_get


def render_timeline(data):

    st.title("🕒 Timeline & Chronology")

    predecessors = safe_get(data, "chronology.predecessor_circulars", [])
    amendments = safe_get(data, "chronology.amendments", [])
    context = safe_get(data, "chronology.historical_context", [])

    topic = safe_get(data, "regulation_topic", "Regulation")

    # -----------------------------
    # 🌳 TREE DIAGRAM (IMPROVED)
    # -----------------------------
    st.subheader("🌳 Policy Timeline Tree")

    dot = graphviz.Digraph()

    # 🔥 GLOBAL GRAPH SETTINGS (CRITICAL FIX)
    dot.attr(
        rankdir='LR',          # Left → Right (timeline feel)
        size='12,6',           # Bigger canvas
        nodesep='1.0',         # Space between nodes
        ranksep='1.5',         # Space between levels
        bgcolor='transparent'
    )

    # 🔥 NODE STYLE (Dark UI Friendly)
    dot.attr(
        'node',
        shape='box',
        style='rounded,filled',
        fontname='Helvetica',
        fontsize='10',
        fontcolor='white',
        color='#00C2FF',
        fillcolor='#0E1117'
    )

    # 🔥 EDGE STYLE
    dot.attr(
        'edge',
        color='#888888',
        arrowsize='0.7'
    )

    # Root node (highlighted)
    dot.node("root", f"📜 {topic}", fillcolor="#1F77B4", color="#1F77B4")

    # Function to safely truncate text
    def truncate(text, max_len=80):
        return text[:max_len] + "..." if len(text) > max_len else text

    # Section colors
    section_colors = {
        "Predecessor Circulars": "#6A5ACD",
        "Amendments": "#FF7F50",
        "Historical Context": "#2E8B57"
    }

    # Add sections
    def add_section(section_name, items):

        if not items:
            return

        section_id = section_name.lower().replace(" ", "_")

        # Section node
        dot.node(
            section_id,
            f"📂 {section_name}",
            fillcolor=section_colors.get(section_name, "#333333"),
            color=section_colors.get(section_name, "#333333")
        )

        dot.edge("root", section_id)

        # Child nodes
        for i, item in enumerate(items):
            node_id = f"{section_id}_{i}"

            dot.node(
                node_id,
                truncate(item),
                fillcolor="#111827",
                color="#444"
            )

            dot.edge(section_id, node_id)

    add_section("Predecessor Circulars", predecessors)
    add_section("Amendments", amendments)
    add_section("Historical Context", context)

    # 🔥 IMPORTANT: Use container width
    st.graphviz_chart(dot, use_container_width=True)

    # -----------------------------
    # 📊 TABLE + ANALYTICS
    # -----------------------------
    timeline_data = []

    for p in predecessors:
        timeline_data.append({"Stage": "Predecessor", "Event": p})

    for a in amendments:
        timeline_data.append({"Stage": "Amendment", "Event": a})

    for c in context:
        timeline_data.append({"Stage": "Context", "Event": c})

    if timeline_data:
        df = pd.DataFrame(timeline_data)

        with st.expander("📋 View Raw Timeline Table"):
            st.dataframe(df, use_container_width=True)

        st.subheader("📈 Event Distribution")
        st.bar_chart(df["Stage"].value_counts())

    else:
        st.warning("No timeline data available")