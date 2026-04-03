# Classification Dataset Audit Summary

- Expected classes: ['bird', 'drone']
- Found classes: ['bird', 'drone']
- Total counted .jpg images: 3319
- Non-.jpg files found: 0

## Split-wise counts

| split   | class_name   |   image_count |
|:--------|:-------------|--------------:|
| train   | bird         |          1414 |
| train   | drone        |          1248 |
| valid   | bird         |           217 |
| valid   | drone        |           225 |
| test    | bird         |           121 |
| test    | drone        |            94 |

# Detection Dataset Audit Summary

## Split Counts

| split   | image_dir_exists   | label_dir_exists   |   image_count |   label_count |
|:--------|:-------------------|:-------------------|--------------:|--------------:|
| train   | True               | True               |          2728 |          2728 |
| valid   | True               | True               |           448 |           448 |
| test    | True               | True               |           224 |           224 |

## Validation Summary

- Total validation rows: 4940
- Invalid rows: 81


## Notes
- Binary classes confirmed as Bird and Drone if folder names match exactly.
- All critical outputs are saved under Drive-backed folders.
