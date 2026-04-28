# 🎓 Education Regulation Impact Analyzer (ERIA)

An AI-powered platform that simplifies complex education policies, regulations, and circulars into clear, structured, and stakeholder-friendly insights.

## 🚀 Live Demo
🔗 http://13.233.94.140:8501

---

## 🚀 Project Overview

Education regulations issued by bodies like UGC, AICTE, NAAC, and the Ministry of Education are often complex and difficult to interpret.

**ERIA solves this problem by:**

* Extracting content from regulation documents (PDF/HTML)
* Using AI (LLMs) to analyze and simplify content
* Generating structured insights including:

  * 📌 Summary (simple language)
  * 👥 Stakeholder impact
  * ⚠️ Risks & challenges
  * 📊 Short/Medium/Long-term impact
  * 🕒 Policy timeline (optional)

---

## 🧠 System Architecture

```
PDF Input
↓
Text Extraction (PyMuPDF)
↓
Preprocessing (Cleaning + Structuring)
↓
Layer 2 → Topic Classification (Hugging Face - BART MNLI)
↓
Layer 1 → LLM Analysis (Gemini API)
↓
Structured JSON Output
↓
Streamlit Dashboard
```

---

## 🏗️ Project Structure

```
ERIA/
│
├── config/                # Configuration files (paths, model settings)
├── data/                  # Raw and processed data
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── artifacts/             # Intermediate outputs (chunks, embeddings)
├── reports/               # Final analysis outputs
├── logs/                  # Application logs
├── docs/                  # Documentation
├── notebooks/             # Development notebooks
├── scripts/               # CLI scripts
│
├── src/
│   ├── ingestion/         # PDF extraction
│   ├── preprocessing/     # Cleaning & chunking
│   ├── llm/               # LLM integration & prompts
│   ├── analysis/          # Core analysis modules
│   └── utils/             # Helpers & configs
│
├── app/                   # Streamlit UI
├── tests/                 # Unit tests
├── deployment/            # Deployment configs
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ERIA.git
cd ERIA
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate environment
Windows
```bash
venv\Scripts\activate
```

Linux/Mac
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```


## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_api_key_here
GROQ_API_KEY=your_backup_key_here
MODEL_NAME=gemini-1.5-flash
```

---

## ▶️ Running the Application

### Run Streamlit App

```bash
streamlit run app/app.py
```

---

## 🧩 Core Components

### 🔹 Layer 1 — LLM (Gemini API)

Handles:

* Summarization
* Stakeholder analysis
* Risk detection
* Impact forecasting

---

### 🔹 Layer 2 — NLP Model (Hugging Face)

Used for:

* Regulation topic classification

Model:

* `facebook/bart-large-mnli`

---

## 📊 Outputs

The system generates:

* 📄 Plain-language summary
* 👥 Stakeholder impact report
* ⚠️ Risk analysis
* 📈 Impact timeline
* 🗂️ Structured JSON output

---

## 🧪 Testing

Run tests:

```bash
pytest tests/
```

---

## 🚀 Deployment

You can deploy using:

* Streamlit Cloud
* Hugging Face Spaces
* Docker (optional)

---

## 📌 Future Enhancements

* 🔍 Semantic search across policies
* 🧠 Knowledge graph for regulation mapping
* 📊 Advanced visualization dashboards
* 📚 Multi-document comparison

---

## 📜 License

This project is for educational purposes.

---

## 🙌 Acknowledgements

* Hugging Face Transformers
* Google Gemini API
* Streamlit
* PyMuPDF

---

## 💡 Author

SHUBHAM MASAL
AI/ML Engineer | NLP Enthusiast

