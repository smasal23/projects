# EcoType — Forest Cover Type Prediction (Supervised ML)

Predict the **forest cover type (7 classes)** from cartographic, topographic, and soil-related features using supervised machine learning, focusing on strong generalization and reproducible training/evaluation.

---

## Project Summary & Objectives

This project builds an end-to-end **multi-class classification** pipeline to predict forest cover type. It covers data ingestion, preprocessing, baseline modeling, tuning, evaluation using robust metrics, and deployment via a lightweight Streamlit app.

**Objectives**
- Build a clean, reproducible ML workflow (configs + pipelines + artifacts).
- Achieve strong predictive performance measured via **Accuracy** and **Macro F1**.
- Provide maintainable code and clear documentation.
- Package inference into a simple Streamlit UI for quick testing/demo.

---

## Dataset Source

This project uses the **Covertype / Forest Cover Type** dataset (commonly available via):
- **UCI Machine Learning Repository**: “Covertype Data Set”
- **Kaggle**: “Forest Cover Type Prediction”

> Place the raw dataset file(s) under: `data/raw/`  
> (Example: `data/raw/covtype.csv` or equivalent format)

---

## Project Workflow

1. **Setup & Configuration**
   - Define paths and training settings in `configs/`.
2. **Data Loading**
   - Read raw dataset from `data/raw/`.
3. **Preprocessing**
   - Train/test split, scaling/encoding as required, pipeline creation.
4. **Baseline Modeling**
   - Train baseline classifiers and log metrics.
5. **Hyperparameter Tuning**
   - Run CV-based search with **Macro F1** as the primary objective.
6. **Final Evaluation**
   - Evaluate on holdout test set; generate metrics and reports.
7. **Model Packaging**
   - Save best pipeline/model and metadata for consistent inference.
8. **Deployment (Streamlit)**
   - Load saved model and serve predictions in a minimal UI.

---

## Evaluation Metrics

We evaluate model performance using:

- **Accuracy**: Overall correctness across all classes.
- **Macro F1**: Average F1-score across classes, giving **equal weight to each class** (important if class distribution is imbalanced).

> Primary metric for tuning: **Macro F1**  
> Secondary metric for reporting: **Accuracy**
---

## Repository Structure
```
EcoType/
│
├── app/                      # Streamlit app (later update)
│   └── app.py
│
├── configs/                  # Central config files
│   ├── paths.yaml
│   ├── train.yaml
│   └── model_grid.yaml
│
├── data/
│   ├── raw/                  # Original dataset (immutable)
│   ├── interim/              # Intermediate transformed data
│   └── processed/            # Final dataset for modeling
│
├── models/                   # Saved models/pipelines
│
├── notebooks/                # EDA / experiments / prototyping
│
├── reports/                  # Metrics, plots, summaries
│
├── src/                      # Core source code
│   ├── data/                 # data loading/processing modules
│   ├── features/             # feature engineering
│   ├── models/               # training/tuning/inference code
│   └── utils/                # helpers (logging, IO, etc.)
│
├── requirements.txt
├── environment.yml           # optional
└── README.md
```
---

## Tech Stack

- Python 3.10+ (recommended)
- NumPy, Pandas
- Scikit-learn
- (Optional) XGBoost / LightGBM (if used in your experiments)
- Matplotlib / Seaborn (for EDA and plots)
- Streamlit (for app deployment)
- Joblib (model persistence)


## Conclusion

EcoType provides a complete, reproducible workflow to predict forest cover types using supervised learning. The project emphasizes clean structure, robust evaluation (Accuracy + Macro F1), and a clear path to deployment via Streamlit. With saved pipelines and configuration-driven training, it can be extended easily with new models, improved feature engineering, and better UI/UX for real-world usage.