# Transfer Learning Report

## Overview

This phase compares multiple pretrained backbones for aerial object classification using a two-stage strategy: frozen-head training followed by optional fine-tuning of the upper backbone layers.

## Best Model So Far

- Best model: **efficientnetb0**
- Validation accuracy: **0.9864**
- Test accuracy: **0.9814**
- Validation F1-score: **0.9867**
- Training time (seconds): **4291.91**

## Model Trade-Offs

### efficientnetb0
- Backbone: efficientnetb0
- Validation accuracy: 0.9864
- Test accuracy: 0.9814
- Training time (seconds): 4291.91
- Speed bucket: medium
- Deployment note: Reasonably lightweight with a good balance of efficiency and accuracy.

### resnet50
- Backbone: resnet50
- Validation accuracy: 0.9661
- Test accuracy: 0.9581
- Training time (seconds): 8201.83
- Speed bucket: slower
- Deployment note: Stronger feature extractor but heavier for edge deployment.

### mobilenet
- Backbone: mobilenetv2
- Validation accuracy: 0.9457
- Test accuracy: 0.9488
- Training time (seconds): 1695.24
- Speed bucket: fast
- Deployment note: Highly deployment-friendly and lightweight.

## Comparison Table

| model_name     | backbone       |   val_accuracy |   val_f1_score |   test_accuracy |   test_f1_score |   training_time_seconds | speed_bucket   | deployment_note                                                        | best_model_path                                                                                                     |
|:---------------|:---------------|---------------:|---------------:|----------------:|----------------:|------------------------:|:---------------|:-----------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| efficientnetb0 | efficientnetb0 |       0.986425 |       0.986667 |        0.981395 |        0.978723 |                 4291.91 | medium         | Reasonably lightweight with a good balance of efficiency and accuracy. | /content/drive/MyDrive/Aerial_Object_Classification_Detection/models/classification/efficientnetb0/best_model.keras |
| resnet50       | resnet50       |       0.966063 |       0.967603 |        0.95814  |        0.953368 |                 8201.83 | slower         | Stronger feature extractor but heavier for edge deployment.            | /content/drive/MyDrive/Aerial_Object_Classification_Detection/models/classification/resnet50/best_model.keras       |
| mobilenet      | mobilenetv2    |       0.945701 |       0.948718 |        0.948837 |        0.944162 |                 1695.24 | fast           | Highly deployment-friendly and lightweight.                            | /content/drive/MyDrive/Aerial_Object_Classification_Detection/models/classification/mobilenet/best_model.keras      |
