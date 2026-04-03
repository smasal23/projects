# Model Comparison Notes

- Best transfer model selected for final evaluation: **efficientnetb0**
- Optional second transfer model included: **resnet50**
- Final selected classifier after full test-set evaluation: **resnet50**

## Selection Criteria

- best balanced performance on the held-out test set
- stable generalization relative to validation performance
- manageable model size
- practical inference suitability for Streamlit deployment

## Practical Deployment Recommendation

- Recommended deployment classifier: **resnet50**
- Reason: Heavier backbone; good accuracy potential but slower deployment.

Criteria:
- Accuracy
- Generalization gap
- Model size
- Inference speed

Final model selected based on balanced performance.