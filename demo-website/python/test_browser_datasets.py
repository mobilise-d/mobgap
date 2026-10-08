"""File-backed upload selection and metadata contracts."""

from pathlib import Path

import pandas as pd
import pytest
from mobgap.data import load_mobilised_matlab_format
from scipy.io import loadmat, savemat

from browser_datasets import UploadedMatlabDataset

ROOT = Path(__file__).resolve().parents[2]
HA = ROOT / "example_data/data/lab/HA/001"


def test_each_index_row_loads_its_trial_and_survives_clone():
    expected = load_mobilised_matlab_format(HA / "data.mat")
    dataset = UploadedMatlabDataset(HA / "data.mat", metadata_path=HA / "infoForAlgo.mat")
    assert set(dataset.index_as_tuples()) == set(expected)
    for datapoint in dataset:
        selected = datapoint.clone()
        assert selected.selected_data_file == HA / "data.mat"
        pd.testing.assert_frame_equal(selected.data_ss, expected[tuple(selected.group_label)].imu_data["LowerBack"])
        assert selected.participant_metadata == pytest.approx({"height_m": 1.59, "sensor_height_m": 0.964})


def test_manual_metadata_survives_subsetting_without_changing_file_defaults():
    dataset = UploadedMatlabDataset(HA / "data.mat", metadata_path=HA / "infoForAlgo.mat")
    overridden = (
        dataset.clone()
        .set_params(
            participant_metadata_override={"height_m": 1.8, "sensor_height_m": 1.0, "cohort": "MS"},
            measurement_condition="free_living",
        )
        .get_subset(test="Test11")
    )
    assert overridden.participant_metadata == {"height_m": 1.8, "sensor_height_m": 1.0, "cohort": "MS"}
    assert overridden.recording_metadata["measurement_condition"] == "free_living"
    assert dataset[0].participant_metadata["height_m"] == 1.59


def test_two_level_self_contained_file_exposes_recording_index_and_partial_metadata(tmp_path):
    original = loadmat(HA / "data.mat", squeeze_me=True, struct_as_record=False, mat_dtype=True)
    path = tmp_path / "free-living.mat"
    savemat(
        path,
        {
            "data": {"TimeMeasure1": {"Recording1": original["data"].TimeMeasure1.Test11.Trial1}},
            "infoForAlgo": {"TimeMeasure1": {"SensorHeight": 96.4}},
        },
    )
    dataset = UploadedMatlabDataset(path, metadata_path=path, measurement_condition="free_living")
    assert dataset.index.to_dict("records") == [{"time_measure": "TimeMeasure1", "recording": "Recording1"}]
    assert len(dataset.data_ss) == 13759
    assert dataset.participant_metadata == pytest.approx({"sensor_height_m": 0.964})
