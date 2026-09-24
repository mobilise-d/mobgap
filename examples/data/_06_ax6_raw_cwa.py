# ruff: noqa: D205
"""
Load and split a raw CWA recording
==================================

``AX6Dataset`` reads CWA files directly through the Rust reader
``cwa_reader_rs``.
This avoids converting recordings to CSV. On Python 3.10 or newer, install
the optional reader with ``pip install mobgap[ax6]``.

The reader can load a selected time window without reading the full recording.
Pass the dataset a DataFrame with full UTC ``start_time`` and ``end_time``
timestamps for each window, or a function that builds this table from recording
metadata. The built-in splitters cover fixed frequencies, UTC days, and hours.

The small example file comes from Open Movement. See its
`source and license notes <https://github.com/mobilise-d/mobgap/blob/main/example_data/data/ax6/README.md>`_.
"""

# %%
from functools import partial

import pandas as pd
from mobgap.data import (
    AX6Dataset,
    get_example_cwa_data_path,
    split_at_frequency,
)

common_options = {
    "path": get_example_cwa_data_path(),
    "participant_metadata": {
        "height_m": 1.7,
        "sensor_height_m": 1.0,
        "cohort": "HA",
    },
    "recording_metadata": {"measurement_condition": "free_living"},
}

# %%
# Split at fixed UTC boundaries. A module-level function with ``partial`` is
# serializable by tpcp and joblib, so a worker can clone the dataset.
by_five_minutes = AX6Dataset(
    **common_options,
    splitter=partial(split_at_frequency, frequency="5min", label="five_minute"),
)

# %%
# Select a window and read its sensor-frame data.
first_window = by_five_minutes.get_subset(recording="five_minute_1")
first_window.data_ss.head()

# %%
# A manually created test list is just a table of named time intervals.
# Here two short test windows start at known offsets into the example recording.
start = by_five_minutes.index.iloc[0].start_time
test_list = pd.DataFrame(
    {
        "test": ["walk_1", "walk_2"],
        "start_time": [start, start + pd.Timedelta(minutes=2)],
        "end_time": [
            start + pd.Timedelta(seconds=20),
            start + pd.Timedelta(minutes=2, seconds=20),
        ],
    }
)
by_test = AX6Dataset(**common_options, splitter=test_list)

# %%
by_test.get_subset(test="walk_2").data_ss.head()
