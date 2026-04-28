# ERIA System Architecture

## Overview

ERIA is designed as a modular NLP + LLM pipeline for analyzing education regulations.

## Pipeline Flow

PDF Input  
→ Text Extraction (PyMuPDF)  
→ Preprocessing (cleaning + structuring)  
→ Topic Classification (Hugging Face BART MNLI)  
→ LLM Analysis (Gemini API)  
→ Structured JSON Output  
→ Streamlit UI  

## Key Design Principles

- Modular architecture
- Single LLM call (prompt-driven)
- Separation of concerns
- Config-driven design
- Scalable and maintainable

## Layers

### Layer 0 — Data Processing
- Ingestion
- Preprocessing

### Layer 2 — NLP Model
- Classification using Hugging Face

### Layer 1 — LLM Core
- Summarization
- Impact analysis
- Risk detection

### UI Layer
- Streamlit dashboard
