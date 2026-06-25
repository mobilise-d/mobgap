import json

import pandas as pd


def load_json_to_dataframe_flat(json_file):
    """
    Load JSON with window-level features and convert to DataFrame.

    Parameters
    ----------
    json_file : str
        Path to JSON file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per 5s window, columns include:
        - id
        - sensor
        - segment_id
        - window_center_idx
        - label
        - all features from features dict
    """
    # Open JSON
    with open(json_file) as f:
        data = json.load(f)

    rows = []
    for window_entry in data:
        row = {
            "id": window_entry["id"],
            "sensor": window_entry.get("sensor"),
            "segment_id": window_entry["segment_id"],
            "window_center_idx": window_entry["window_center_idx"],
            "label": window_entry["label"],
        }
        # Flatten features dict into row
        row.update(window_entry["features"])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
