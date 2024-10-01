"""Load and reformat the original GSD results of all algorithms for the TVS dataset.

This does the following steps:
1. Extract the participant id and the algorithm name from the file name.
2. Load the matlab file and extract the start and end of all GSD in all trials and time measures.
3. Recode the participant id to the new ids in line with the published TVS dataset.
4. Save the results in one json file per algorithm.
"""
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import mat_struct
from tqdm import tqdm

from mobgap.data._mobilised_matlab_loader import _parse_matlab_struct, _parse_until_test_level

ROOT_DATA_PATH = Path("../../../mobgap_validation/")

PARTICIPANT_ID_MAPPING = Path("../../../mobilised_tvs_data/TVS-participant-22032024.csv")
# The new ids are the first integer of the participantid concatenated with the newly generated ids.
# This way we keep the information about the original about the original recording site.
id_mapping = (
    pd.read_csv(PARTICIPANT_ID_MAPPING)
    .astype(str)
    .assign(new_id=lambda df_: df_.participantid.str[0].str.cat(df_.id.str.rjust(3, "0")))
    .set_index("participantid")["new_id"]
    .to_dict()
)

ALGO_NAME_MAPPING = {}


def parse_single_test(data: mat_struct) -> pd.DataFrame:
    """Parse the data of a single test from the GSD output."""
    data = data.SU.LowerBack.GSD

    if isinstance(data, mat_struct):
        data = [data]

    dps = [_parse_matlab_struct(dp) for dp in data]
    if dps:
        sampling_rate = dps[0]["GSD_fs"]
        return pd.DataFrame.from_records(dps).drop(columns="GSD_fs").rename(columns=lambda s: s.lower()).mul(sampling_rate).astype(int)
    return pd.DataFrame(columns=["start", "end"]).astype({"start": int, "end": int})



def process(path: Path) -> tuple[str, Optional[pd.DataFrame]]:
    path = path.resolve()
    cohort = path.parent.parent.name
    condition = path.parent.name.lower().replace("-", "_")
    participant_id, *_, algo_name = path.name.split("_", 2)
    algo_name = algo_name.rsplit("_", 1)[0]
    new_participant_id = id_mapping[participant_id]
    new_algo_name = ALGO_NAME_MAPPING.get(algo_name, algo_name)

    data = loadmat(str(path), squeeze_me=True, struct_as_record=False, mat_dtype=True)
    data_per_test = _parse_until_test_level(data["GSD_Output"], ("SU",))
    per_file_results = {}
    for test, test_data in data_per_test:
        identifier = (new_algo_name, cohort, new_participant_id, *test)
        per_file_results[identifier] = parse_single_test(test_data)

    if not per_file_results:
        return condition , None
    col_names = ["algorithm", "cohort", "participant_id"]
    if condition == "Free-living":
        col_names.extend(["time_measure", "recording"])
    else:
        col_names.extend(["time_measure", "test", "trial"])
    return condition, pd.concat(per_file_results, names=[*col_names, "wb_id"])

# %%
GSD_DATA_PATH = ROOT_DATA_PATH / "_old_data_raw/GSD"
assert GSD_DATA_PATH.exists(), GSD_DATA_PATH.resolve()
all_results = {}

for path in tqdm(list(GSD_DATA_PATH.rglob("*.mat"))):
    condition, result = process(path)
    if result is not None:
        all_results.setdefault(condition, []).append(result)

for condition, result in all_results.items():
    out = pd.concat(result).sort_index()
    out_dir = ROOT_DATA_PATH / "data/gsd" / condition
    out_dir.mkdir(exist_ok=True)
    for name, group in tqdm(list(out.groupby("algorithm"))):
        group.reset_index("algorithm", drop=True).to_json(out_dir / f"{name}.zip", orient="index", indent=2)

