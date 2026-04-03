# Custom CNN Report


## Experiment Summary

- Classes: bird, drone
- Input size: 224 x 224
- Objective: binary aerial object classification
- Model family: custom CNN baseline

## Architecture Choice

A custom CNN with 4 convolution blocks was selected to provide a clear, explainable baseline before deeper transfer-learning experiments. The architecture uses convolution + batch normalization + ReLU + max pooling in each block, followed by dropout and a dense classifier head. This balances representational capacity with moderate regularization for a relatively small two-class image dataset.

## Augmentation Choice

The training pipeline applies rotation, horizontal flipping, optional vertical flipping only when visually sensible, zoom, brightness variation, and translation-based spatial shifts. These augmentations are intended to improve robustness to orientation changes, mild viewpoint variation, and lighting differences commonly seen in aerial imagery.

## Early Observations

- Review training vs validation accuracy and loss curves.
- Check whether validation loss diverges while training loss continues to fall.
- If train accuracy rises much faster than validation accuracy, reduce capacity or increase regularization.
- If both curves remain low, consider more epochs, architecture tuning, or stronger feature extraction.

## Key Saved Artifacts

- `models/classification/custom_cnn/best_model.keras`
- `models/classification/custom_cnn/final_model.keras`
- `models/classification/custom_cnn/history.json`
- `models/classification/custom_cnn/metrics.json`
- `figures/preprocessing/augmentation_preview.png`
- `figures/training/custom_cnn_accuracy.png`
- `figures/training/custom_cnn_loss.png`

## Validation Metrics

- Accuracy: 0.7805
- Precision: 0.7883
- Recall: 0.7778
- F1-score: 0.7830

## Test Metrics

- Accuracy: 0.8512
- Precision: 0.8780
- Recall: 0.7660
- F1-score: 0.8182