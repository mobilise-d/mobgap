"""Create some example data from the full T1 CVS data set.

This extracts the first two participants from the full data set and adds some random noise to the data.
The data is still re-identifiable via the wb_id and the data, but as the actual DMO values are modified, it should no
relevant information about the participants should be leaked.
"""

import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

# Replace these with the actual paths to the data
in_file = "/home/arne/Downloads/cvs-T1-wb-dmo-27-11-2023.csv"
mapping_file = "/home/arne/Downloads/study-instances-Cohort Site-2023-08-08h22m09s48.csv"

out_file = HERE / "cvs-T1-test_data.csv"
out_file_mapping = HERE / "cvs-T1-test_data_mapping.csv"

test_data = pd.read_csv(in_file, nrows=4753)
mapping = pd.read_csv(mapping_file)
# We drop the cohort columns, as they might reveal the condition of the participants
mapping = mapping.drop(columns=["Measurement.Group", *mapping.columns[mapping.columns.str.startswith("Cohort")]])
# Then we get the participants that are in the test data
test_participants = test_data["participantid"].unique()
mapping = mapping[mapping["Local.Participant"].isin(test_participants)]

p_id_offset = random.randint(0, 10)

test_data["participantid"] += p_id_offset
mapping["Local.Participant"] += p_id_offset
for c in [
    "duration",
    "averagestridelength",
    "averagestrideduration",
    "averagestepduration_so",
    "averagecadence",
    "averagestridespeed",
    "averagestepduration_so",
]:
    test_data[c] *= 1 + random.random() / 10
test_data.to_csv(out_file, index=False)
mapping.to_csv(out_file_mapping, index=False)
