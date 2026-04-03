# Phase 2 Preprocessing Validation Summary

## Dataset root
- /content/drive/MyDrive/Aerial_Object_Classification_Detection/data/processed/classification

## Class mappings
- class_to_index: {'bird': 0, 'drone': 1}
- index_to_class: {0: 'bird', 1: 'drone'}

## Split counts

| split   |   image_count |
|:--------|--------------:|
| test    |           215 |
| train   |          2662 |
| valid   |           442 |

## Batch checks
- Train batch shape: (32, 3, 224, 224)
- Valid batch shape: (32, 3, 224, 224)
- Test batch shape: (32, 3, 224, 224)
- Train tensor min/max: (0.0000, 1.0000)

## Saved figures
- /content/drive/MyDrive/Aerial_Object_Classification_Detection/figures/preprocessing/resized_preview.png
- /content/drive/MyDrive/Aerial_Object_Classification_Detection/figures/preprocessing/normalized_preview.png