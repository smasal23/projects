# YOLOv8 Detection Report

## Environment

- Project root: `/content/drive/MyDrive/Aerial_Object_Classification_Detection`
- Detection dataset root: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/data/raw/object_detection_dataset`
- YOLO model size: `yolov8n.pt`
- Epochs: 25
- Image size: 640
- Batch size: 16

## Dataset Validation

- Validation dataframe rows: 4940
- Invalid rows found: 0
- Class names: ['bird', 'drone']

## Training Artifacts

- Training run directory: `/content/runs/detect/models/detection/yolov8`
- Best checkpoint: `/content/runs/detect/models/detection/yolov8/weights/best.pt`
- Last checkpoint: `/content/runs/detect/models/detection/yolov8/weights/last.pt`
- Final exported detector: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/models/detection/final/best_detector.pt`
- Results CSV: `/content/runs/detect/models/detection/yolov8/results.csv`
- Args YAML: `/content/runs/detect/models/detection/yolov8/args.yaml`

## Validation Metrics

- Precision: 0.856368359331025
- Recall: 0.7921080508474576
- mAP@50: 0.8216003262763782
- mAP@50-95: 0.5353635509862287

## Saved Figures

- Overlay preview: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/figures/detection/dataset_overlay_preview.png`
- Validation preview: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/figures/detection/yolo_val_preview.png`
- Test predictions: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/figures/detection/yolo_test_predictions.png`
- New image predictions: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/figures/detection/yolo_new_image_predictions.png`
- Docs prediction image: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/docs/images/yolo_predictions.png`

## Qualitative Notes

# YOLOv8 Inference Observations

## Test-set inference

- Source: `/content/drive/MyDrive/Aerial_Object_Classification_Detection/data/raw/object_detection_dataset/test/images`
- Saved predictions directory: `/content/runs/detect/models/detection/yolov8_test_predictions`
- Number of result items: 224

The detector-generated images should be reviewed for box tightness, missed detections, false positives, and class consistency under scale and background variation.