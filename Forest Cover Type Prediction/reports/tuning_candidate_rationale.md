# Tuning Candidate Selection Rationale

## Selected Models

- random_forest
- extra_trees

## Selection Basis

- Primary metric: cross-validated macro F1
- Secondary metric: cross-validated accuracy
- Stability considered through standard deviation across CV folds

## Model Comparison Summary

- **random_forest** ranked first with macro F1 = 0.911450 (std = 0.004864) and accuracy = 0.959613.
- **extra_trees** ranked second with macro F1 = 0.911337 (std = 0.004756) and accuracy = 0.954404.

## Why these models move to tuning

- Both models clearly outperform the remaining baselines on macro F1.
- Both are stable across CV folds, with low standard deviation.
- Both are strong ensemble methods for structured tabular multiclass classification.
- Their performance gap is small enough that tuning could change final ranking.

## Why other models were not selected

- Decision tree was materially below the top two and less competitive overall.
- Gradient boosting underperformed the top ensemble models by a clear margin.
- Logistic regression and SVC showed poor macro F1 and minority-class handling.
- GaussianNB was not competitive for this feature space.

## Recommendation

- Move **random_forest** and **extra_trees** to the tuning phase.
- Use macro F1 as the primary tuning metric.