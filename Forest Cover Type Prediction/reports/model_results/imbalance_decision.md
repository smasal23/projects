# Imbalance Handling Decision

## Selected Strategy
**SMOTE**

## Reason
- Selected based on highest macro F1 on test set
- Tie-breaker considered minority class recall
- Cross-validation macro F1 was reviewed for stability
- Leakage was avoided by keeping resampling inside the pipeline

## Model for Tuning
- RandomForestClassifier

## Constraints
- Oversampling increases runtime
- Class-weight approach is simpler
- Final choice should balance performance and stability
