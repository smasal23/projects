# Data Dictionary — EcoType (Forest Cover Type Prediction)

This document describes the **input features** and **target** used in the EcoType project.
It is designed to make the dataset easy to understand for:
- readers of your GitHub repository,
- future-you during model iteration,
- and anyone trying to reproduce or extend your work.

> Dataset: Covertype / Forest Cover Type  
> Typical raw file: `data/raw/covtype.csv` (may vary)

---

## 1) Target Variable

### `Cover_Type`
**Meaning:** The forest cover type class label (multi-class classification).  
**Type:** Categorical (integer-encoded)  
**Typical values:** `1` to `7`

**Common mapping (dataset standard):**
1. Spruce/Fir  
2. Lodgepole Pine  
3. Ponderosa Pine  
4. Cottonwood/Willow  
5. Aspen  
6. Douglas-fir  
7. Krummholz  

> Note: If your dataset uses different labels or mapping, update this section accordingly.

---

## 2) Feature Columns (Typical Covertype Schema)

> The Covertype dataset usually includes **54 features**:
- 10 continuous/numeric features
- 4 binary wilderness area indicators
- 40 binary soil type indicators

### A) Continuous / Numeric Features

| Column | Description | Units / Range (typical) |
|---|---|---|
| `Elevation` | Elevation above sea level | meters; approx 1850–3850 |
| `Aspect` | Aspect (direction slope faces) | degrees; 0–360 |
| `Slope` | Slope steepness | degrees; 0–60 (approx) |
| `Horizontal_Distance_To_Hydrology` | Horizontal distance to nearest surface water features | meters; 0+ |
| `Vertical_Distance_To_Hydrology` | Vertical distance to nearest surface water features | meters; can be negative/positive |
| `Horizontal_Distance_To_Roadways` | Horizontal distance to nearest roadway | meters; 0+ |
| `Hillshade_9am` | Hillshade index at 9am | 0–255 |
| `Hillshade_Noon` | Hillshade index at noon | 0–255 |
| `Hillshade_3pm` | Hillshade index at 3pm | 0–255 |
| `Horizontal_Distance_To_Fire_Points` | Horizontal distance to nearest wildfire ignition points | meters; 0+ |

> Ranges above are common for the standard dataset; your actual ranges may differ slightly.

---

### B) Wilderness Area Indicators (Binary)

Exactly one of these is typically `1` for a given row (one-hot representation).

| Column | Description | Values |
|---|---|---|
| `Wilderness_Area1` | Wilderness area category 1 | 0/1 |
| `Wilderness_Area2` | Wilderness area category 2 | 0/1 |
| `Wilderness_Area3` | Wilderness area category 3 | 0/1 |
| `Wilderness_Area4` | Wilderness area category 4 | 0/1 |

---

### C) Soil Type Indicators (Binary)

The dataset usually includes **40 soil type columns**: `Soil_Type1` ... `Soil_Type40`.  
Exactly one soil type is typically `1` for a given row (one-hot representation).

| Column Pattern | Description | Values |
|---|---|---|
| `Soil_Type1` ... `Soil_Type40` | Soil type category indicators | 0/1 |

---

## 3) Data Quality Notes

### Missing Values
- The standard Covertype dataset is commonly distributed **without missing values**.
- If your version has missing values, document:
  - which columns are affected,
  - how you handle them (drop/impute).

### Duplicate Rows
- Check for duplicates and decide whether to keep or remove them based on your modeling strategy.

### Class Distribution
- Cover types are often **imbalanced**.
- Use **Macro F1** to avoid ignoring minority classes.

---

## 4) Derived / Engineered Features (Add Later)

Add any engineered features here once created, including:
- feature name,
- formula/logic,
- why it helps,
- whether it introduces leakage risk.

Example template (replace later):

| Derived Feature | How it’s computed | Why it’s useful | Leakage risk? |
|---|---|---|---|
| `elevation_slope_interaction` | `Elevation * Slope` | captures terrain interaction | Low |
| `distance_ratio_road_hydro` | `Horizontal_Distance_To_Roadways / (Horizontal_Distance_To_Hydrology + 1)` | relative proximity signal | Low |

---

## 5) Final Checklist (Before Publishing)

- [ ] Confirm your raw dataset column names match this dictionary  
- [ ] Confirm target encoding / label mapping  
- [ ] Add any dataset-specific quirks (missing values, renamed columns, etc.)  
- [ ] Update derived feature section as feature engineering evolves
