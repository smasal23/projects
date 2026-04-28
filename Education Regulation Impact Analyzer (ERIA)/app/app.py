import streamlit as st
import sys
import os
import tempfile
import json
import hashlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.pipeline import ERIAPipeline

from app.components.summary_view import render_summary
from app.components.stakeholder_view import render_stakeholders
from app.components.risk_view import render_risks
from app.components.impact_view import render_impact
from app.components.timeline_view import render_timeline
from app.utils.pdf_export import generate_pdf

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="ERIA Dashboard", layout="wide")

# -----------------------
# LOAD CSS
# -----------------------
def load_css():
    with open("app/assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("📊 ERIA Dashboard")

view = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Upload & Analyze",
        "Summary",
        "Stakeholders",
        "Risks",
        "Impact",
        "Timeline"
    ]
)

st.sidebar.markdown("""
**Education Regulation Impact Analyzer (ERIA)**

🔍 Extracts structured insights from policy documents using LLMs.

---

### 🧭 Navigation
- Upload & Analyze → Run pipeline
- Summary → Key insights
- Stakeholders → Who is affected
- Risks → Positives & risks
- Impact → Time-based effects
- Timeline → Regulation history

---

### 👨‍💻 Author
Shubham Masal

### ⚙️ Version
v1.0 (Phase 7 UI)

### 📌 Status
Prototype → Moving to Production
""")

# -----------------------
# SESSION STATE
# -----------------------
if "data" not in st.session_state:
    st.session_state.data = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

# -----------------------
# PIPELINE INIT
# -----------------------
pipeline = ERIAPipeline()

# -----------------------
# HOME PAGE
# -----------------------
if view == "Home":

    st.title("📊 Education Regulation Impact Analyzer (ERIA)")

    st.markdown("""
    ---

    ## 🔍 What is ERIA?

    ERIA is an AI-powered system that converts **unstructured regulation PDFs**
    into **structured, decision-ready insights**.

    Instead of reading 100+ pages manually, ERIA gives:

    ✔ Regulation Summary
    ✔ Stakeholder Impact
    ✔ Risk Signals
    ✔ Timeline Analysis
    ✔ Structured JSON output

    ---

    ## ⚙️ How It Works

    PDF → Text Extraction → Preprocessing → Chunking → Smart Filtering → LLM Analysis → Structured Output → Dashboard

    ---

    ## 🚀 How to Use

    1. Go to **Upload & Analyze**
    2. Upload a regulation PDF
    3. Wait for processing (10–30 sec)
    4. Navigate through tabs:
       - Summary
       - Stakeholders
       - Risks
       - Impact
       - Timeline

    ---

    ## ⚠️ Important Notes

    - Large PDFs may take longer
    - Very short documents may give limited insights
    - LLM responses depend on input quality
    - Rate limits may temporarily block processing

    ---

    ## 🎯 Use Cases

    - Policy Analysis
    - Education Regulation Review
    - Compliance Monitoring
    - Research & Reports

    ---

    ## 🧠 What Makes This Different?

    ✔ Full pipeline (not just LLM call)
    ✔ Structured schema output
    ✔ Multi-view dashboard
    ✔ Real-world regulation use case

    ---

    ## 👨‍💻 Built By

    **Shubham Masal**

    ---
    """)


# -----------------------
# UPLOAD PAGE
# -----------------------
if view == "Upload & Analyze":

    st.title("📄 Upload Regulation PDF")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:

        file_bytes = uploaded_file.read()

        # ✅ Create hash of file
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # Initialize
        if "file_hash" not in st.session_state:
            st.session_state.file_hash = None

        # ✅ Process ONLY if new file
        if file_hash != st.session_state.file_hash:

            with st.spinner("Processing document..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    temp_path = tmp.name

                progress = st.progress(0)
                status = st.empty()

                try:
                    status.text("📥 Ingesting document...")
                    progress.progress(10)
                    ingest_output = pipeline.ingest(temp_path)

                    progress.progress(20)
                    status.text("💾 Saving intermediate data...")

                    from src.utils.config import BASE_DIR
                    from src.utils.helpers import save_json

                    interim_path = os.path.join(BASE_DIR, "data", "interim", "temp.json")
                    save_json(ingest_output, interim_path)

                    progress.progress(35)
                    status.text("🧹 Preprocessing & chunking...")
                    processed = pipeline.preprocess(interim_path)
                    chunks = processed["chunks"]

                    progress.progress(55)
                    status.text("🧠 Preparing LLM input...")
                    llm_input = pipeline.build_llm_input(chunks)

                    progress.progress(75)
                    status.text("🤖 Running LLM analysis...")
                    llm_output = pipeline.run_llm(llm_input)

                    progress.progress(90)
                    status.text("📊 Finalizing insights...")
                    final_output = pipeline.post_process(llm_output)

                    progress.progress(100)
                    status.text("✅ Done!")

                    # ✅ Save everything
                    st.session_state.data = final_output
                    st.session_state.file_hash = file_hash

                    # ✅ Reset PDF cache
                    if "pdf_bytes" in st.session_state:
                        del st.session_state.pdf_bytes

                    st.success("🎉 Analysis Completed!")

                except Exception as e:
                    st.error(f"❌ Pipeline failed: {str(e)}")
                    st.stop()

        else:
            st.info("⚡ Using cached analysis (no reprocessing)")

        # -----------------------
        # DOWNLOAD OPTIONS (SAFE)
        # -----------------------
        if st.session_state.data:

            data = st.session_state.data

            st.download_button(
                label="📥 Download JSON Output",
                data=json.dumps(data, indent=2),
                file_name="eria_analysis.json",
                mime="application/json"
            )

            # ✅ Generate PDF ONCE (cached in memory)
            if "pdf_bytes" not in st.session_state:
                pdf_path = generate_pdf(data)
                with open(pdf_path, "rb") as f:
                    st.session_state.pdf_bytes = f.read()

            st.download_button(
                label="📄 Download PDF Report",
                data=st.session_state.pdf_bytes,
                file_name="eria_report.pdf",
                mime="application/pdf"
            )

            # Debug
            with st.expander("🔍 Debug Output"):
                st.json(data)

# -----------------------
# DISPLAY DATA
# -----------------------
data = st.session_state.data

if data:

    st.sidebar.success("Data Loaded")

    raw = data.get("raw", data)  # fallback safety

    if view == "Summary":
        render_summary(raw)

    elif view == "Stakeholders":
        render_stakeholders(raw)

    elif view == "Risks":
        render_risks(raw)

    elif view == "Impact":
        render_impact(raw)

    elif view == "Timeline":
        render_timeline(raw)

else:
    if view not in ["Home", "Upload & Analyze"]:
        st.warning("⚠️ Please upload and analyze a document first.")