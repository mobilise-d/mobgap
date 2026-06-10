"""
In order to train and evaluate the XGBost model you need to create a feature dataset for wear-time detection.

Feature extraction is performed using the utility functions in:
    src/mobgap/weartime/utils/feature_extraction.py

To generate the full feature set (230 features), call:
    features = extract_full_features(win)
    label = is_wear  # 1 = wear, 0 = non-wear; requires continuous sample-level reference data


To generate the reduced feature set (79 features, 90% SHAP importance), call:
    features = extract_features_90pct(win)
    label = is_wear  # 1 = wear, 0 = non-wear; requires continuous sample-level reference data

Each window entry should be saved in the following structure:
    {
        "id": id,
        "sensor": sensor,
        "segment_id": segment_id,
        "window_center_idx": window_center_idx,
        "label": label,
        "features": features
    }

This structure is consumed directly by load_json_to_dataframe_flat
in the training and evaluation scripts.

Environment variables:
    WEARTIME_RAW_DATA_DIR:   Path to raw IMU data directory
    WEARTIME_REFERENCE_FILE: Path to reference JSON file
"""
