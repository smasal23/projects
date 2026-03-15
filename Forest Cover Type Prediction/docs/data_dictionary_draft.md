# Draft Data Dictionary

    This is the first draft of the data dictionary generated from the interim dataset.

    ## Dataset Summary
    - Rows: 145890
    - Columns: 13
    - Target Column: `cover_type`

    ## Columns
    | column_name                        | dtype   | is_target   |   null_count | example_value   |
|:-----------------------------------|:--------|:------------|-------------:|:----------------|
| elevation                          | int64   | no          |            0 | 2596            |
| aspect                             | int64   | no          |            0 | 51              |
| slope                              | int64   | no          |            0 | 3               |
| horizontal_distance_to_hydrology   | int64   | no          |            0 | 258             |
| vertical_distance_to_hydrology     | int64   | no          |            0 | 0               |
| horizontal_distance_to_roadways    | int64   | no          |            0 | 510             |
| hillshade_9am                      | int64   | no          |            0 | 221             |
| hillshade_noon                     | int64   | no          |            0 | 232             |
| hillshade_3pm                      | int64   | no          |            0 | 148             |
| horizontal_distance_to_fire_points | int64   | no          |            0 | 6279            |
| cover_type                         | object  | yes         |            0 | Aspen           |
| wilderness_area                    | int64   | no          |            0 | 1               |
| soil_type                          | int64   | no          |            0 | 29              |

    ## Notes
    - Descriptions will be refined later using the official project brief.
    - Units/Ranges and derived features can be added in later phases.
    