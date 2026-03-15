# Data Split Summary

    **Generated on:** 2026-03-12 23:20:41

    ## Files
    - Input dataset: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\data\interim\forest_cover_cleaned.csv`
    - Saved outputs:
      - `data/processed/X_train.csv`
      - `data/processed/X_test.csv`
      - `data/processed/y_train.csv`
      - `data/processed/y_test.csv`

    ## Split Configuration
    - Target column: `cover_type`
    - Split type: **Train/Test**
    - Test size: **20%**
    - Train size: **80%**
    - Random state: **42**
    - Stratification: **Enabled**

    ## Dataset Shape Before Split
    - Total rows: **145890**
    - Total columns: **13**

    ## Output Shapes
    - `X_train`: **116712 rows × 12 columns**
    - `X_test`: **29178 rows × 12 columns**
    - `y_train`: **116712 rows**
    - `y_test`: **29178 rows**

    ## No Overlap Check
    - Train rows checked: **116712**
    - Test rows checked: **29178**
    - Overlap count: **0**
    - Overlap found: **False**

    ## Full Dataset Target Distribution
    | cover_type        |   count |   percentage |
|:------------------|--------:|-------------:|
| Aspen             |    3069 |         2.1  |
| Cottonwood/Willow |    2160 |         1.48 |
| Douglas-fir       |    2160 |         1.48 |
| Krummholz         |    2160 |         1.48 |
| Lodgepole Pine    |  103071 |        70.65 |
| Ponderosa Pine    |    2160 |         1.48 |
| Spruce/Fir        |   31110 |        21.32 |

    ## Training Set Target Distribution
    | cover_type        |   count |   percentage |
|:------------------|--------:|-------------:|
| Aspen             |    2455 |         2.1  |
| Cottonwood/Willow |    1728 |         1.48 |
| Douglas-fir       |    1728 |         1.48 |
| Krummholz         |    1728 |         1.48 |
| Lodgepole Pine    |   82457 |        70.65 |
| Ponderosa Pine    |    1728 |         1.48 |
| Spruce/Fir        |   24888 |        21.32 |

    ## Test Set Target Distribution
    | cover_type        |   count |   percentage |
|:------------------|--------:|-------------:|
| Aspen             |     614 |         2.1  |
| Cottonwood/Willow |     432 |         1.48 |
| Douglas-fir       |     432 |         1.48 |
| Krummholz         |     432 |         1.48 |
| Lodgepole Pine    |   20614 |        70.65 |
| Ponderosa Pine    |     432 |         1.48 |
| Spruce/Fir        |    6222 |        21.32 |

    ## Interpretation
    - Stratified splitting was applied using `cover_type`.
    - This helps preserve class proportions in both training and test sets.
    - No overlap should exist between training and test samples.
    - The saved split files are ready for feature engineering and model training.
    