# Project Overview — EcoType (Forest Cover Type Prediction)

## 1) Project Goal

Build a complete, reproducible **multi-class classification** system that predicts **forest cover type (7 classes)** from cartographic/topographic/soil features, and package the best-performing model into a simple **Streamlit** app for demo/inference.

---

## 2) Problem Statement

Given environmental and geographic features (elevation, slope, hydrology distances, roadways distance, hillshades, soil and wilderness indicators, etc.), predict the **Cover_Type** label (1–7). The focus is on **generalization**, **clean ML pipelines**, and **reproducible experiments**.

---

## 3) Key Deliverables

### A) Code & Reproducibility
- [ ] Clean repo structure (`src/`, `configs/`, `docs/`, `reports/`, `models/`, `app/`)
- [ ] Config-driven training (`configs/paths.yaml`, `configs/train.yaml`, `configs/model_grid.yaml`)
- [ ] Reusable preprocessing + modeling pipeline (sklearn `Pipeline`)
- [ ] Scripted training + evaluation (`src/train.py`, `src/evaluate.py`) or notebook equivalents

### B) Data & Modeling Outputs
- [ ] Processed dataset saved to `data/processed/`
- [ ] Baseline model results (at least 2–3 models)
- [ ] Tuned model results (Grid/Random Search CV)
- [ ] Final model artifact saved (e.g., `models/best_model.joblib`)
- [ ] Reports: metrics JSON, classification report text, confusion matrix plot

### C) Documentation
- [ ] `README.md` (setup, run, results, app usage)
- [ ] `docs/project_overview.md` (this doc)
- [ ] `docs/data_dictionary.md` (features + target description)

### D) Deployment (Demo)
- [ ] Streamlit app to load model and run predictions
- [ ] Minimal UI + input validation
- [ ] Clear instructions to run app locally

---

## 4) 11-Day Timeline (Day 1–11)

> This timeline is designed to be **strict and execution-focused**.  
> Outcome by Day 11: a trained + tuned model, complete reports, and a working Streamlit demo.

### Day 1 — Setup & Repo Foundations
**Objectives**
- Create folder structure, configs skeleton, README skeleton
- Lock environment + dependencies

**Tasks**
- Create: `configs/paths.yaml`, `configs/train.yaml`, `configs/model_grid.yaml`
- Add: `requirements.txt`
- Add documentation placeholders: `docs/project_overview.md`, `docs/data_dictionary.md`

**Exit Criteria**
- Repo runs locally without path issues
- Config files exist and are readable

---

### Day 2 — Data Ingestion & Validation
**Objectives**
- Load dataset from `data/raw/`
- Validate columns, types, missing values, target classes

**Tasks**
- Write data loader utility (`src/data/load_data.py`)
- Basic checks: duplicates, missingness, invalid values, class distribution

**Exit Criteria**
- A clean dataframe is produced reliably from raw data
- A short validation summary saved to `reports/`

---

### Day 3 — EDA (Core Understanding)
**Objectives**
- Understand feature distributions and relationships to target

**Tasks**
- Feature distributions (numerical + binary indicators)
- Target distribution
- Quick sanity plots (optional): elevation vs cover type, slope vs cover type

**Exit Criteria**
- EDA notebook/report with key insights
- Clear list of preprocessing needs (scaling yes/no; handling binary columns)

---

### Day 4 — Preprocessing Pipeline (No Leakage)
**Objectives**
- Build sklearn preprocessing pipeline that is safe and reproducible

**Tasks**
- Identify numeric vs binary/indicator columns
- Build `ColumnTransformer` + `Pipeline`
- Train/test split with stratification

**Exit Criteria**
- Pipeline runs end-to-end on train split
- No leakage (fit only on train)

---

### Day 5 — Baseline Modeling
**Objectives**
- Establish baseline performance and pick top 1–2 candidates for tuning

**Tasks**
- Baselines: Logistic Regression, Random Forest, Gradient Boosting (or similar)
- Evaluate on validation / CV
- Record: Accuracy + Macro F1

**Exit Criteria**
- Baseline metrics saved in `reports/metrics.json` (or separate baseline file)
- Candidate models selected for tuning

---

### Day 6 — Hyperparameter Tuning (Small Grid First)
**Objectives**
- Tune selected models using CV with primary metric = Macro F1

**Tasks**
- Implement GridSearchCV or RandomizedSearchCV
- Use config `configs/model_grid.yaml`
- Save best estimator and params

**Exit Criteria**
- Best tuned model saved to `models/`
- Best params saved to report

---

### Day 7 — Final Evaluation & Reporting
**Objectives**
- Evaluate tuned model on holdout test set and generate full report

**Tasks**
- Classification report (precision/recall/F1 per class)
- Confusion matrix plot
- Save metrics + artifacts

**Exit Criteria**
- `reports/classification_report.txt`
- `reports/figures/confusion_matrix.png`
- Updated `reports/metrics.json`

---

### Day 8 — Error Analysis & Improvements
**Objectives**
- Understand failure modes and apply 1–2 targeted improvements

**Examples**
- Feature scaling adjustments (if needed)
- Try one additional model (e.g., ExtraTrees) OR expand grid slightly
- Handle class imbalance (optional, only if needed)

**Exit Criteria**
- Documented improvement attempt
- Decision made: final model selected

---

### Day 9 — Packaging & Inference-Ready Pipeline
**Objectives**
- Make inference stable and user-safe

**Tasks**
- Ensure pipeline includes preprocessing + model together
- Add input schema checks / column ordering consistency
- Save final model as a single artifact

**Exit Criteria**
- `models/best_model.joblib` loads and predicts on new rows without errors

---

### Day 10 — Streamlit App (Demo)
**Objectives**
- Build a minimal but robust UI for predictions

**Tasks**
- Create `app/app.py`
- Load model from `models/`
- Create input widgets for key numeric features
- Display prediction output (+ probabilities if available)

**Exit Criteria**
- `streamlit run app/app.py` works locally
- App produces predictions reliably

---

### Day 11 — Final Polish & GitHub Publish
**Objectives**
- Make the repo “public-ready”

**Tasks**
- Finalize README: setup, run, results, app usage
- Add screenshots (optional): confusion matrix, app UI
- Clean code formatting + remove unused files
- Add LICENSE + final checks

**Exit Criteria**
- Repo runs clean from scratch (fresh environment)
- Docs + results are clear and complete

---

## 5) ML Lifecycle Mapping (Data → Deploy)

### Step 1: Data
- Raw dataset placed in `data/raw/`
- Loader reads from `configs/paths.yaml`

### Step 2: Validation
- Basic checks (nulls, duplicates, class distribution)
- Output quick report to `reports/`

### Step 3: Preprocessing
- Train/test split with stratification
- `ColumnTransformer` handles:
  - numeric scaling (if used)
  - passthrough binary columns
- Pipeline prevents data leakage

### Step 4: Modeling
- Baseline models
- Compare using: Accuracy + Macro F1

### Step 5: Tuning
- CV strategy: StratifiedKFold
- Primary metric: Macro F1
- Small grid initially → expand if needed

### Step 6: Evaluation
- Holdout test set metrics
- Confusion matrix + classification report

### Step 7: Packaging
- Save end-to-end pipeline (`joblib`)
- Save config snapshots and metrics

### Step 8: Deployment
- Streamlit app loads the saved model
- User enters features → predicted cover type displayed

---

## 6) Risks & Mitigations

### Risk 1: Data leakage (inflated performance)
**Mitigation**
- Always use sklearn `Pipeline`
- Fit preprocessing only on training split
- CV should wrap the entire pipeline

### Risk 2: Wrong column schema during inference
**Mitigation**
- Save the full pipeline (preprocess + model together)
- Enforce a fixed feature list/order in inference
- Add checks in Streamlit before prediction

### Risk 3: Class imbalance hurting minority classes
**Mitigation**
- Use **Macro F1** as primary metric
- Inspect per-class F1
- Optionally use class_weight / resampling only if needed

### Risk 4: Over-tuning / long runtimes
**Mitigation**
- Start with small grids
- Prefer RandomizedSearch when grids get large
- Use sensible CV folds (5) and early narrowing

### Risk 5: Non-reproducible runs (different results each time)
**Mitigation**
- Fix random seeds in split + CV + models
- Track params and configs in repo

### Risk 6: App breaks due to missing model file
**Mitigation**
- Standardize artifact path in `configs/paths.yaml`
- Add clear “Train first” messaging in app
- Provide example model artifact instructions

---

## 7) Definition of Done (DoD)

A run is considered complete when:
- `python src/train.py --config configs/train.yaml` produces a saved model
- `python src/evaluate.py --config configs/train.yaml` produces metrics + plots
- `streamlit run app/app.py` works and returns predictions
- README and docs are complete and accurate
- Repo is publish-ready on GitHub
