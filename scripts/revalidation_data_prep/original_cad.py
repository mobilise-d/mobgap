"""Load and reformat the original cadence results of the selected algorithms for the TVS dataset.

This does the following steps:
1. Extract the participant id and the algorithm name from the file name.
2. Load the matlab file and extract the cadence in all trials and time measures.
3. Recode the participant id to the new ids in line with the published TVS dataset.
4. Save the results in one json file per algorithm.

"""

from pathlib import Path
from typing import Optional, TypedDict

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


def parse_single_test(data: mat_struct) -> pd.DataFrame:
    """Parse the data of a single test from the GSD output."""
    data = data.SU.LowerBack.CAD

    if isinstance(data, mat_struct):
        data = [data]

    dps = [_parse_matlab_struct(dp) for dp in data]
    if dps:
        df = pd.DataFrame.from_records(dps)
        if len(df) > 0:
            return (
                df[["start", "stop", "cadSec", "cadMean"]]
                .rename({"cadSec": "cad_per_sec", "cadMean": "cad_mean", "stop": "end"}, axis=1)
                # For cadence the start and the end values are provided in seconds for some reason.
                .assign(
                    cad_per_sec=lambda df_: df_.cad_per_sec.apply(
                        lambda x: [x] if isinstance(x, float) else x.tolist()
                    ),
                    # We adjust the start to be 0-based (from matlab 1 based)
                    # Note, that we don't adjust the end, because we also want the last sample to be inclusive.
                    # This is valid here, even though the values are calculated from the second values.
                    # We align with the way the values were used in the matlab code.
                    start=lambda df_: as_samples(df_.start, DATA_SAMPLING_RATE) - 1,
                    end=lambda df_: as_samples(df_.end, DATA_SAMPLING_RATE),
                )
                .astype({"start": int, "end": int, "cad_per_sec": object, "cad_mean": float})
            )

    return pd.DataFrame(columns=["start", "end", "cad_per_sec", "cad_mean"]).astype(
        {"start": int, "end": int, "cad_per_sec": object, "cad_mean": float}
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
    algo_name = algo_name.split("-CAD_")[1].rsplit("_", 1)[0]
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
    data_per_test = _parse_until_test_level(data["CAD_Output"], ("SU",))
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
CAD_DATA_PATH = ROOT_DATA_PATH / "_old_data_raw/CAD"

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
    out_dir = ROOT_DATA_PATH / "_extracted_results/cad" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, group in tqdm(list(out.groupby("algorithm"))):
        group = group.reset_index("algorithm", drop=True)
        group.reset_index().to_csv(out_dir / f"{name}.csv", index=False)
