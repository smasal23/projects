# Feature Selection Summary

## Removed Constant Features

- None

## Removed Near-Constant Features

- None

## Removed Highly Correlated Features

- hillshade_noon
- hillshade_range
- horizontal_distance_to_hydrology
- slope
- vertical_distance_to_hydrology_abs

## Baseline Feature Importance (Top 20)

| Feature | Importance |
|---|---:|
| elevation | 0.216129 |
| soil_type | 0.123954 |
| wilderness_area | 0.089329 |
| horizontal_distance_to_roadways | 0.087716 |
| horizontal_distance_to_fire_points | 0.067316 |
| road_fire_ratio | 0.044865 |
| road_fire_gap | 0.044093 |
| hydrology_road_ratio | 0.039932 |
| hydrology_fire_ratio | 0.033890 |
| hydrology_distance | 0.033606 |
| aspect_cos | 0.029282 |
| elevation_slope_ratio | 0.027270 |
| hillshade_9am | 0.026375 |
| vertical_distance_to_hydrology | 0.025629 |
| hillshade_std | 0.023556 |
| aspect | 0.019014 |
| elevation_slope_interaction | 0.018683 |
| hillshade_mean | 0.017680 |
| hillshade_3pm | 0.017079 |
| aspect_sin | 0.014603 |

## Final Selected Features

- elevation
- aspect
- vertical_distance_to_hydrology
- horizontal_distance_to_roadways
- hillshade_9am
- hillshade_3pm
- horizontal_distance_to_fire_points
- wilderness_area
- soil_type
- aspect_sin
- aspect_cos
- hydrology_distance
- hillshade_mean
- hillshade_std
- elevation_slope_interaction
- elevation_slope_ratio
- road_fire_gap
- road_fire_ratio
- hydrology_fire_ratio
- hydrology_road_ratio