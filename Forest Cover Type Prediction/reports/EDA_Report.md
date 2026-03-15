# EDA Report — EcoType Forest Cover Classification

## 1. Dataset Overview

- Source file: `F:\DATA SCIENCE\Projects\Forest Cover Type Prediction\data\processed\Forest_cover_engineered.csv`
- Shape: `145890` rows × `26` columns
- Target column: `cover_type`
- Numeric features: `25`
- Categorical features: `0`
- Missing cells total: `0`
- Duplicate rows: `0`

## 2. Class Distribution

- Class imbalance ratio (max/min count): `47.72`

| class             |   count |   percentage |
|:------------------|--------:|-------------:|
| Aspen             |    3069 |         2.1  |
| Cottonwood/Willow |    2160 |         1.48 |
| Douglas-fir       |    2160 |         1.48 |
| Krummholz         |    2160 |         1.48 |
| Lodgepole Pine    |  103071 |        70.65 |
| Ponderosa Pine    |    2160 |         1.48 |
| Spruce/Fir        |   31110 |        21.32 |

![Class Distribution](figures/eda/class_distribution_bar.png)

## 3. Summary Statistics

Top rows from numeric summary statistics:

| feature                            |   count |            mean |              std |    min |           25% |        median |           50% |          75% |            max |        skew |     kurtosis |
|:-----------------------------------|--------:|----------------:|-----------------:|-------:|--------------:|--------------:|--------------:|-------------:|---------------:|------------:|-------------:|
| elevation_slope_ratio              |  145890 |     4.86088e+06 |      1.1893e+08  |   43.5 |   187.375     |   263.909     |   263.909     |   388.625    |      3.379e+09 |  24.501     |   600.099    |
| road_fire_ratio                    |  145890 | 92859.3         |      1.81587e+07 |    0   |     0.676161  |     1.06611   |     1.06611   |     1.7449   |      4.818e+09 | 244.117     | 62643.5      |
| hydrology_fire_ratio               |  145890 |  9000.07        |      1.53274e+06 |    0   |     0.0338003 |     0.0799334 |     0.0799334 |     0.157735 |      3.9e+08   | 193.874     | 41127.9      |
| hydrology_road_ratio               |  145890 |  1795.99        | 437621           |    0   |     0.0330944 |     0.0701873 |     0.0701873 |     0.125995 |      1.24e+08  | 261.164     | 69757.7      |
| elevation_slope_interaction        |  145890 | 34072.9         |  17596.1         |    0   | 21189         | 31966.5       | 31966.5       | 44175        | 166896         |   0.811074  |     1.06562  |
| horizontal_distance_to_fire_points |  145890 |  3044.96        |   1761.88        |    0   |  1608         |  2713         |  2713         |  4478        |   7173         |   0.365249  |    -0.967014 |
| horizontal_distance_to_roadways    |  145890 |  3313.83        |   1687.78        |    0   |  1848         |  3420         |  3420         |  4673        |   7117         |  -0.0582396 |    -1.06688  |
| road_fire_gap                      |  145890 |  1619.41        |   1321.93        |    0   |   566         |  1183.5       |  1183.5       |  2481        |   6017         |   0.868522  |    -0.250112 |
| elevation                          |  145890 |  2874.46        |    210.801       | 1863   |  2747         |  2909         |  2909         |  3004        |   3849         |  -0.664497  |     1.60863  |
| hydrology_distance                 |  145890 |   255.918       |    194.578       |    0   |    98.2344    |   216.668     |   216.668     |   368.069    |   1356.94      |   0.973905  |     0.862716 |
| horizontal_distance_to_hydrology   |  145890 |   251.825       |    192.474       |    0   |    95         |   212         |   212         |   362        |   1343         |   0.984953  |     0.876187 |
| aspect                             |  145890 |   141.127       |    107.719       |    0   |    54         |   108         |   108         |   217        |    360         |   0.689432  |    -0.817218 |

## 4. Correlation Analysis

- Strong correlation pairs found (|r| >= 0.75): `13`

| feature_1                        | feature_2                          |   correlation |   abs_correlation |
|:---------------------------------|:-----------------------------------|--------------:|------------------:|
| horizontal_distance_to_hydrology | hydrology_distance                 |      0.999412 |          0.999412 |
| hillshade_range                  | hillshade_std                      |      0.992695 |          0.992695 |
| hillshade_noon                   | hillshade_mean                     |      0.985755 |          0.985755 |
| slope                            | elevation_slope_interaction        |      0.981452 |          0.981452 |
| vertical_distance_to_hydrology   | vertical_distance_to_hydrology_abs |      0.963914 |          0.963914 |
| hillshade_3pm                    | hillshade_std                      |     -0.872872 |          0.872872 |
| hillshade_3pm                    | hillshade_range                    |     -0.867466 |          0.867466 |
| hillshade_3pm                    | aspect_sin                         |     -0.836382 |          0.836382 |
| hillshade_9am                    | hillshade_std                      |      0.797403 |          0.797403 |
| hillshade_9am                    | hillshade_3pm                      |     -0.788674 |          0.788674 |
| hillshade_9am                    | aspect_sin                         |      0.78767  |          0.78767  |
| hillshade_9am                    | hillshade_range                    |      0.772372 |          0.772372 |

![Correlation Heatmap](figures/eda/correlation_heatmap.png)

## 5. Target vs Top Features

- Variance-based top features chosen for visual inspection: `['elevation_slope_ratio', 'road_fire_ratio', 'hydrology_fire_ratio', 'hydrology_road_ratio', 'elevation_slope_interaction', 'horizontal_distance_to_fire_points']`

![target_vs_elevation_slope_ratio_boxplot](figures/eda/target_vs_elevation_slope_ratio_boxplot.png)
![target_vs_road_fire_ratio_boxplot](figures/eda/target_vs_road_fire_ratio_boxplot.png)
![target_vs_hydrology_fire_ratio_boxplot](figures/eda/target_vs_hydrology_fire_ratio_boxplot.png)
![target_vs_hydrology_road_ratio_boxplot](figures/eda/target_vs_hydrology_road_ratio_boxplot.png)

## 6. Outlier Impact Notes

- Outliers were reviewed using the IQR rule.
- In this terrain dataset, outliers may represent valid rare terrain conditions rather than data errors.
- Outlier-heavy features should be interpreted carefully during model selection and scaling decisions.

| features                           |           q1 |          q3 |         iqr |   outlier_count |   outlier_pct |
|:-----------------------------------|-------------:|------------:|------------:|----------------:|--------------:|
| road_fire_ratio                    |    0.676161  |    1.7449   |   1.06874   |           13927 |          9.55 |
| hydrology_road_ratio               |    0.0330944 |    0.125995 |   0.0929002 |           13032 |          8.93 |
| elevation_slope_ratio              |  187.375     |  388.625    | 201.25      |           11947 |          8.19 |
| hydrology_fire_ratio               |    0.0338003 |    0.157735 |   0.123935  |           11341 |          7.77 |
| vertical_distance_to_hydrology     |    7         |   51        |  44         |            7123 |          4.88 |
| vertical_distance_to_hydrology_abs |    9         |   52        |  43         |            6969 |          4.78 |
| hillshade_9am                      |  207         |  232        |  25         |            5242 |          3.59 |
| elevation                          | 2747         | 3004        | 257         |            4448 |          3.05 |
| hillshade_mean                     |  188         |  202        |  14         |            4135 |          2.83 |
| hillshade_range                    |   72         |  111        |  39         |            3411 |          2.34 |

![road_fire_ratio_outlier_boxplot](figures/eda/road_fire_ratio_outlier_boxplot.png)
![hydrology_road_ratio_outlier_boxplot](figures/eda/hydrology_road_ratio_outlier_boxplot.png)
![elevation_slope_ratio_outlier_boxplot](figures/eda/elevation_slope_ratio_outlier_boxplot.png)
![hydrology_fire_ratio_outlier_boxplot](figures/eda/hydrology_fire_ratio_outlier_boxplot.png)

## 7. Leakage Suspicion Checklist

- Check identifier risk: `road_fire_ratio` has very high uniqueness ratio (0.99).
- Verify all engineered features were created before train/test split only when they do not use target statistics.
- If any encoding, aggregation, or scaling used full-dataset information, revisit to avoid train-test contamination.
- Wilderness_Area_* and Soil_Type_* style one-hot columns are expected predictors, not leakage by default.

## 8. EDA Conclusions

- Review class balance before final metric choice and model weighting.
- Monitor highly correlated features during model training.
- Keep outlier-heavy terrain variables under observation instead of dropping them blindly.
- Recheck engineered columns for train-test contamination before preprocessing/modeling.
- Use this EDA as the bridge into the preprocessing and modeling phases.
