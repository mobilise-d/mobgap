"""A script to prepare the TVS dataset for release.

Important: After running this script, you should run the two matlab scripts in the `scripts` folder to corretly remove
the turn data and fix some of the metadata in the infoForAlgo files.
"""

import shutil
from pathlib import Path

import pandas as pd

from mobgap.data._mobilised_tvs_dataset import BaseTVSDataset, TVSFreeLivingDataset, TVSLabDataset


def _generate_concurrent_validity_report(dataset: BaseTVSDataset, outpath: Path) -> None:
    """Compare the data from the infoForAlgo file to the information from the clinical data.

    This is only use

    """
    clinical_info = dataset.participant_information
    participant_metadata = dataset.participant_metadata_as_df
    walking_aid_col = (
        "walking_aid_used_laboratory"
        if dataset._MEASUREMENT_CONDITION == "Laboratory"
        else "walking_aid_used_free_living"
    )
    clinical_info = clinical_info.assign(walking_aid_used=clinical_info[walking_aid_col].notnull())

    common_cols = clinical_info.columns.intersection(participant_metadata.columns)
    diff = clinical_info[common_cols] != participant_metadata[common_cols]
    # We mask the diff to only show the differences
    diff_clinical_info = clinical_info[common_cols].where(diff).dropna(how="all")
    diff_participant_metadata = participant_metadata[common_cols].where(diff).dropna(how="all")
    pd.concat({"clinical_info": diff_clinical_info, "info_for_algo": diff_participant_metadata}, axis=1).to_csv(outpath)


# %%
TVS_DATA_PATH = Path("../../../mobilised_tvs_data/data_original")
PARTICIPANT_ID_MAPPING = Path("../../../mobilised_tvs_data/TVS-participant-22032024.csv")
OUTPATH = Path("../../../mobilised_tvs_data/tvs_dataset")
USE_MOVE = True

# The new ids are the first integer of the participantid concatenated with the newly generated ids.
# This way we keep the information about the original about the original recording site.
id_mapping = (
    pd.read_csv(PARTICIPANT_ID_MAPPING)
    .astype(str)
    .assign(new_id=lambda df_: df_.participantid.str[0].str.cat(df_.id.str.rjust(3, "0")))
    .set_index("participantid")["new_id"]
    .to_dict()
)
print(id_mapping)

# %%
# We need to do two things:
# 1. Copy all the files and rename the top level folder to the new id.
# 2. Update the participant information to use the new id.

# Let's start with 2.
OUTPATH.mkdir(exist_ok=True, parents=True)

clinical_info = TVS_DATA_PATH / "participant_information.xlsx"
clinical_info_out = OUTPATH / "participant_information.xlsx"

# Load the participant information
# We need to do this manually with openpyxl, to avoid messing up the formatting
# Load the workbook and select the sheet
import openpyxl

workbook = openpyxl.load_workbook(clinical_info)

# Read a specific column (e.g., column 'B')
for sheet_name in ["Participant Characteristics", "Data Quality Summary"]:
    sheet = workbook[sheet_name]
    column_data = []
    for cell in sheet["A"]:
        column_data.append(id_mapping.get(str(cell.value), cell.value))

    for index, value in enumerate(column_data):
        sheet.cell(row=index + 1, column=1).value = value

# Save the workbook with a new name to preserve the original file
workbook.save(clinical_info_out)

# %%
# Now we need to copy the files
# We need to copy the files and rename the participant id to the new id
input("Copying and potentially deleting standardized data\n. Press enter to continue")
for cohort in TVS_DATA_PATH.iterdir():
    if not cohort.is_dir():
        continue

    cohort_out = OUTPATH / cohort.name
    cohort_out.mkdir(exist_ok=True, parents=True)

    for participant in cohort.iterdir():
        if not participant.is_dir():
            continue

        participant_id = id_mapping.get(participant.name)
        participant_out = cohort_out / participant_id

        shutil.rmtree(participant_out, ignore_errors=True)
        if USE_MOVE:
            shutil.move(participant, participant_out)
        else:
            shutil.copytree(participant, participant_out)


# %%
# Generate reports

ds = TVSLabDataset(OUTPATH, missing_reference_error_type="ignore", reference_system="INDIP")
_generate_concurrent_validity_report(ds, PARTICIPANT_ID_MAPPING.parent / "concurrent_validity_report_lab.csv")

ds = TVSFreeLivingDataset(OUTPATH, missing_reference_error_type="ignore", reference_system="INDIP")
_generate_concurrent_validity_report(ds, PARTICIPANT_ID_MAPPING.parent / "concurrent_validity_report_free_living.csv")
