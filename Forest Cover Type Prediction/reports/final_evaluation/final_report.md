# Final Evaluation Report — EcoType Forest Cover Classification

## 1. Final Model
- Selected tuned model: `random_forest`
- Best estimator path: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\artifacts\tuning\random_forest_best_estimator.joblib`
- Best params path: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\artifacts\tuning\random_forest_best_params.yaml`
- CV results path: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\artifacts\tuning\random_forest_cv_results.csv`
- Final refit model path: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\models\random_forest_final_model.joblib`

## 2. Dataset Split
- X_train shape: `(116712, 20)`
- X_test shape: `(29178, 20)`
- y_train shape: `(116712,)`
- y_test shape: `(29178,)`

## 3. Final Test Metrics
- **accuracy**: `0.9594`
- **f1_macro**: `0.9152`
- **f1_weighted**: `0.9590`

## 4. Generalization Check
- CV best macro F1: `0.8949`
- Test macro F1: `0.9152`
- Test accuracy: `0.9594`
- Gap (test - CV): `0.0202`
- Interpretation: Test macro F1 is higher than CV macro F1. Likely favorable split variance.

## 5. Best Hyperparameters
- **model__n_estimators**: `100`
- **model__min_samples_split**: `2`
- **model__min_samples_leaf**: `1`
- **model__max_features**: `sqrt`
- **model__max_depth**: `None`

## 6. Classification Report
|                   |   precision |   recall |   f1-score |      support |
|:------------------|------------:|---------:|-----------:|-------------:|
| Aspen             |    0.919383 | 0.872964 |   0.895572 |   614        |
| Cottonwood/Willow |    0.949309 | 0.953704 |   0.951501 |   432        |
| Douglas-fir       |    0.825054 | 0.884259 |   0.853631 |   432        |
| Krummholz         |    0.934978 | 0.965278 |   0.949886 |   432        |
| Lodgepole Pine    |    0.967014 | 0.985544 |   0.976191 | 20614        |
| Ponderosa Pine    |    0.863208 | 0.847222 |   0.85514  |   432        |
| Spruce/Fir        |    0.956178 | 0.894246 |   0.924176 |  6222        |
| accuracy          |    0.959387 | 0.959387 |   0.959387 |     0.959387 |
| macro avg         |    0.916446 | 0.914745 |   0.915157 | 29178        |
| weighted avg      |    0.959326 | 0.959387 |   0.959041 | 29178        |

## 7. Saved Outputs
- Metrics JSON: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\reports\final_evaluation\metrics.json`
- Classification report CSV: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\reports\final_evaluation\classification_report.csv`
- Confusion matrix CSV: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\reports\final_evaluation\confusion_matrix.csv`
- Confusion matrix PNG: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\reports\figures\final_evaluation\random_forest_confusion_matrix.png`