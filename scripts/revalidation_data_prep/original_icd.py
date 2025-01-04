"""Load and reformat the original ICD (SD) results of the selected algorithms for the TVS dataset.

This does the following steps:
1. Extract the participant id and the algorithm name from the file name.
2. Load the matlab file and extract the initial contacts in all trials and time measures.
3. Convert the initial contacts to samples from the **beginning of the walking bout.**
3. Recode the participant id to the new ids in line with the published TVS dataset.
4. Save the results in one json file per algorithm.

"""

from pathlib import Path
from typing import Optional, TypedDict, Union

import numpy as np
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

ALGO_NAME_MAPPING = {}

DATA_SAMPLING_RATE = 100


def _ic_list_to_samples(ic_list: Union[list[float], float], wb_start: int, sampling_rate: int) -> list[int]:
    if isinstance(ic_list, float):
        ic_list = [ic_list]
    return (np.array(as_samples(ic_list, sampling_rate)) - wb_start).tolist()


def parse_single_test(data: mat_struct) -> pd.DataFrame:
    """Parse the data of a single test from the GSD output."""
    data = data.SU.LowerBack.SD

    if isinstance(data, mat_struct):
        data = [data]

    dps = [_parse_matlab_struct(dp) for dp in data]
    if dps:
        df = pd.DataFrame.from_records(dps)
        if len(df) > 0:
            return (
                df[["GaitSequenceRefined_Start", "GaitSequenceRefined_End", "InitialContact_Event"]]
                .rename(
                    {
                        "InitialContact_Event": "ic_list_rel_to_wb",
                        "cadMean": "cad_mean",
                        "GaitSequenceRefined_End": "end",
                        "GaitSequenceRefined_Start": "start",
                    },
                    axis=1,
                )
                .assign(
                    start=lambda df_: as_samples(df_.start, DATA_SAMPLING_RATE),
                    end=lambda df_: as_samples(df_.end, DATA_SAMPLING_RATE),
                    ic_list_rel_to_wb=lambda df_: (
                        df_.apply(
                            lambda val: _ic_list_to_samples(val.ic_list_rel_to_wb, val.start, DATA_SAMPLING_RATE),
                            axis=1,
                        )
                    ),
                )
            )

    return pd.DataFrame(columns=["start", "end", "ic_list_rel_to_wb"]).astype(
        {"start": int, "end": int, "ic_list_rel_to_wb": object}
    )


class InfoDict(TypedDict):
    cohort: str
    condition: str
    participant_id: str
    algo_name: str
    new_participant_id: str


def get_info(path: Path) -> InfoDict:
    path = path.resolve()
    cohort = path.parent.parent.name
    condition = path.parent.name.lower().replace("-", "_")
    participant_id, *_, algo_name = path.name.split("_", 2)
    algo_name = algo_name.split("-SD_")[1].rsplit("_", 1)[0]
    new_participant_id = id_mapping[participant_id]
    return {
        "cohort": cohort,
        "condition": condition,
        "participant_id": participant_id,
        "algo_name": algo_name,
        "new_participant_id": new_participant_id,
    }


def process(path: Path, info_dict: InfoDict) -> Optional[pd.DataFrame]:
    data = loadmat(str(path), squeeze_me=True, struct_as_record=False, mat_dtype=True)
    data_per_test = _parse_until_test_level(data["SD_Output"], ("SU",))
    per_file_results = {}
    for test, test_data in data_per_test:
        identifier = (info_dict["algo_name"], info_dict["cohort"], info_dict["new_participant_id"], *test)
        per_file_results[identifier] = parse_single_test(test_data)

    if not per_file_results:
        return None
    col_names = ["algorithm", "cohort", "participant_id"]
    if info_dict["condition"] == "free_living":
        col_names.extend(["time_measure", "recording"])
    else:
        col_names.extend(["time_measure", "test", "trial"])
    return pd.concat(per_file_results, names=[*col_names, "wb_id"])


# %%
CAD_DATA_PATH = ROOT_DATA_PATH / "_old_data_raw/SD"

assert CAD_DATA_PATH.exists(), CAD_DATA_PATH.resolve()
all_results = {}

for path in tqdm(list(CAD_DATA_PATH.rglob("*.mat"))):
    file_info = get_info(path)
    result = process(path, file_info)
    if result is not None:
        all_results.setdefault(file_info["condition"], []).append(result)

# %%
for condition, result in all_results.items():
    out = pd.concat(result).sort_index()
    out_dir = ROOT_DATA_PATH / "_extracted_results/icd" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, group in tqdm(list(out.groupby("algorithm"))):
        group = group.reset_index("algorithm", drop=True)
        group.reset_index().to_csv(out_dir / f"{name}.csv", index=False)
