"""Load and reformat the original full pipeline results.

This does the following steps:
1. Extract the participant id from the file name and the cohort fromt he folder.
2. reade the WBASO.mat file and extract the WB-level results.
3. Recode the participant id to the new ids in line with the published TVS dataset.
4. Save the results in a single csv file per condition.

Notes
-----
    - At the moment, we are only extracting the primary results from the WBASO.mat file.
      Other results can be added as needed.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import mat_struct
from tqdm import tqdm

from mobgap.data._mobilised_matlab_loader import _parse_matlab_struct, _parse_until_test_level
from mobgap.utils.conversions import as_samples
from mobgap.utils.misc import get_env_var

ROOT_DATA_PATH = Path(get_env_var("MOBGAP_VALIDATION_DATA_PATH"))
PARTICIPANT_ID_MAPPING = ROOT_DATA_PATH / "_old_data_raw/TVS-participant-22032024.csv"

# The new ids are the first integer of the participantid concatenated with the newly generated ids.
# This way we keep the information about the original about the original recording site.
id_mapping = (
    pd.read_csv(PARTICIPANT_ID_MAPPING)
    .astype(str)
    .assign(new_id=lambda df_: df_.participantid.str[0].str.cat(df_.id.str.rjust(3, "0")))
    .set_index("participantid")["new_id"]
    .to_dict()
)

COL_RENAME_DICT = {
    "Id": "wb_id_orig",
    "Start": "start",
    "End": "end",
    "StartDateTimeUtc": "start_datetime_utc",
    "StartTimestampUtc": "start_timestamp_utc",
    "TimeZone": "time_zone",
    "NumberStrides": "n_strides",
    "Duration": "duration",
    "TerminationReason": "termination_reason",
    "AverageCadence": "avg_cadence",
    "AverageStrideSpeed": "avg_speed",
    "AverageStrideLength": "avg_stride_length",
    "AverageStrideDuration": "avg_stride_duration",
    # "Stride_Start": "stride_start",
    # "Stride_End": "stride_end",
    # "Stride_Duration": "stride_duration",
    # "Stride_Length": "stride_length",
    # "Stride_Speed": "stride_speed",
    # "Stride_Cadence": "stride_cadence",
    # "Stride_LeftRight": "stride_left_right",
    # "InitialContact_Event": "ic_event",
    # "InitialContact_LeftRight": "ic_left_right",
    # "Sec_Time": "sec_time",
    # "Sec_Cadence": "sec_cadence",
    # "Sec_StrideLength": "sec_stride_length",
    # "Sec_Speed": "sec_speed",
}

COL_DTYPES = {
    "wb_id_orig": str,
    "start": int,
    "end": int,
    "start_datetime_utc": str,
    "start_timestamp_utc": int,
    "time_zone": str,
    "n_strides": int,
    "duration": float,
    "termination_reason": str,
    "avg_cadence": float,
    "avg_speed": float,
    "avg_stride_length": float,
    "avg_stride_duration": float,
}

DATA_SAMPLING_RATE = 100


def parse_single_test(data: mat_struct) -> pd.DataFrame:
    """Parse the data of a single test from the GSD output."""
    data = data.SU.LowerBack.MacroWB

    if isinstance(data, mat_struct):
        data = [data]

    dps = [_parse_matlab_struct(dp) for dp in data]
    if dps and "Id" in dps[0]:
        return (
            pd.DataFrame.from_records(dps)[list(COL_RENAME_DICT.keys())]
            .rename(columns=COL_RENAME_DICT)
            .assign(
                start=lambda df_: as_samples(df_.start, DATA_SAMPLING_RATE),
                end=lambda df_: as_samples(df_.end, DATA_SAMPLING_RATE),
            )
            .astype(COL_DTYPES)
        )
    return pd.DataFrame(columns=list(COL_RENAME_DICT.values())).astype(COL_DTYPES)


def process(path: Path) -> tuple[str, Optional[pd.DataFrame]]:
    path = path.resolve()
    cohort = path.parent.parent.name
    condition = path.parent.name.lower().replace("-", "_")
    _, participant_id, *_ = path.name.split("_", 2)
    new_participant_id = id_mapping[participant_id]

    data = loadmat(str(path), squeeze_me=True, struct_as_record=False, mat_dtype=True)
    data_per_test = _parse_until_test_level(data["WBASO_Output"], ("SU",))
    per_file_results = {}
    for test, test_data in data_per_test:
        identifier = (cohort, new_participant_id, *test)
        per_file_results[identifier] = parse_single_test(test_data)

    if not per_file_results:
        return condition, None
    col_names = ["cohort", "participant_id"]
    if condition == "free_living":
        col_names.extend(["time_measure", "recording"])
    else:
        col_names.extend(["time_measure", "test", "trial"])
    return condition, pd.concat(per_file_results, names=[*col_names, "wb_id"])


# %%
RESULT_DATA_PATH = ROOT_DATA_PATH / "_old_data_raw/full_pipeline"
assert RESULT_DATA_PATH.exists(), RESULT_DATA_PATH.resolve()
all_results = {}

for path in tqdm(list(RESULT_DATA_PATH.rglob("*WBASO_Output.mat"))):
    condition, result = process(path)
    if result is not None:
        all_results.setdefault(condition, []).append(result)

# %%
for condition, result in all_results.items():
    out = pd.concat(result).sort_index()
    out_dir = ROOT_DATA_PATH / "_extracted_results/full_pipeline" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(out_dir / "escience_mobilised_pipeline.csv", index=False)
