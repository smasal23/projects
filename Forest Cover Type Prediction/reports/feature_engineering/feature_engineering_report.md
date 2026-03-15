# Feature Engineering Report

## Dataset Shapes

- Original shape: (145890, 13)
- Engineered shape: (145890, 26)
- Number of engineered features: 13

## Engineered Features

- **aspect_sin**: Sine transform of aspect in radians to represent circular direction.
- **aspect_cos**: Cosine transform of aspect in radians to represent circular direction.
- **hydrology_distance**: Euclidean distance to hydrology computed from horizontal and vertical hydrology distances.
- **vertical_distance_to_hydrology_abs**: Absolute vertical distance to hydrology, ignoring above/below sign.
- **hillshade_mean**: Mean hillshade across 9am, noon, and 3pm.
- **hillshade_range**: Difference between maximum and minimum hillshade across the day.
- **hillshade_std**: Standard deviation of hillshade values across the day.
- **elevation_slope_interaction**: Interaction term between elevation and slope.
- **elevation_slope_ratio**: Elevation divided by slope magnitude to capture terrain gradient relationship.
- **road_fire_gap**: Absolute difference between horizontal distance to roadways and fire points.
- **road_fire_ratio**: Horizontal distance to roadways divided by horizontal distance to fire points.
- **hydrology_fire_ratio**: Horizontal distance to hydrology divided by horizontal distance to fire points.
- **hydrology_road_ratio**: Horizontal distance to hydrology divided by horizontal distance to roadways.

## Top 10 Baseline Important Features

- **elevation**: 0.207381
- **soil_type**: 0.125969
- **wilderness_area**: 0.093447
- **horizontal_distance_to_roadways**: 0.082237
- **horizontal_distance_to_fire_points**: 0.065637
- **road_fire_ratio**: 0.042594
- **road_fire_gap**: 0.040717
- **hydrology_road_ratio**: 0.032747
- **hydrology_fire_ratio**: 0.028701
- **aspect_cos**: 0.025306

## Notes

- Baseline feature importance is only an early ranking signal.
- Final feature selection should be validated in the dedicated selection phase.
- Engineered dataset has been saved for downstream modeling.