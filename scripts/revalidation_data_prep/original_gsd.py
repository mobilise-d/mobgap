"""Load and reformat the original GSD results of all algorithms for the TVS dataset.

This does the following steps:
1. Extract the participant id and the algorithm name from the file name.
2. Load the matlab file and extract the start and end of all GSD in all trials and time measures.
3. Recode the participant id to the new ids in line with the published TVS dataset.
4. Save the results in one json file per algorithm.

Notes
-----
- Some algorithm results show GSDs that extend past the end of the data.
  In most cases, these are rounding issues.
  In case of the EPFL_V1-* algorithms, this is caused by an actual bug in the original implementation.
  In both cases, you should clip the end of the GSD to the length of the data to avoid issues during evaluation.
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

ALGO_NAME_MAPPING = {}
DATA_SAMPLING_RATE = 100


def parse_single_test(data: mat_struct) -> pd.DataFrame:
    """Parse the data of a single test from the GSD output."""
    data = data.SU.LowerBack.GSD

    if isinstance(data, mat_struct):
        data = [data]

    dps = [_parse_matlab_struct(dp) for dp in data]
    if dps:
        return (
            pd.DataFrame.from_records(dps)
            .drop(columns="GSD_fs")
            .rename(columns=lambda s: s.lower())
            .pipe(lambda df_: as_samples(df_, DATA_SAMPLING_RATE))
        )
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
        return condition, None
    col_names = ["algorithm", "cohort", "participant_id"]
    if condition == "free_living":
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

# %%
for condition, result in all_results.items():
    out = pd.concat(result).sort_index()
    out_dir = ROOT_DATA_PATH / "_extracted_results/gsd" / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, group in tqdm(list(out.groupby("algorithm"))):
        group = group.reset_index("algorithm", drop=True)
        group.reset_index().to_csv(out_dir / f"{name}.csv", index=False)
