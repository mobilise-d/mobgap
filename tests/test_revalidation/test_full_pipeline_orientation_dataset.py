"""Tests for full-pipeline orientation validation dataset helpers."""

import pandas as pd
from pandas.testing import assert_frame_equal

from mobgap.consts import SF_SENSOR_COLS
from mobgap.data import GaitDatasetFromData
from mobgap.data.base import ReferenceData
from mobgap.re_orientation.pipeline import REORIENTATION_ROTATIONS
from mobgap.utils.conversions import to_body_frame, to_sensor_frame
from mobgap.utils.rotations import flip_dataset
from revalidation.full_pipeline._orientation_dataset import MisorientedDataset


class ReferenceGaitDatasetFromData(GaitDatasetFromData):
    """In-memory gait dataset with minimal reference data for wrapper tests."""

    @property
    def reference_parameters_(self) -> ReferenceData:
        """Return minimal recording-level reference parameters."""
        return ReferenceData(
            wb_list=pd.DataFrame({"start": [0], "end": [2]}),
            ic_list=None,
            turn_parameters=None,
            stride_parameters=None,
        )

    @property
    def reference_parameters_relative_to_wb_(self) -> ReferenceData:
        """Return minimal WB-relative reference parameters."""
        return self.reference_parameters_

    @property
    def reference_sampling_rate_hz_(self) -> float:
        """Return reference sampling rate."""
        return self.sampling_rate_hz


def _base_dataset() -> ReferenceGaitDatasetFromData:
    data = pd.DataFrame(
        {
            "acc_x": [1.0, 2.0, 3.0],
            "acc_y": [4.0, 5.0, 6.0],
            "acc_z": [7.0, 8.0, 9.0],
            "gyr_x": [10.0, 11.0, 12.0],
            "gyr_y": [13.0, 14.0, 15.0],
            "gyr_z": [16.0, 17.0, 18.0],
        }
    )
    return ReferenceGaitDatasetFromData(
        _data={
            ("HA", "001"): {"LowerBack": data},
            ("COPD", "002"): {"LowerBack": data + 100},
        },
        _sampling_rate_hz=100.0,
        _participant_metadata={
            ("HA", "001"): {"cohort": "HA", "height_m": 1.75, "sensor_height_m": 1.0},
            ("COPD", "002"): {"cohort": "COPD", "height_m": 1.65, "sensor_height_m": 0.95},
        },
        _recording_metadata={
            ("HA", "001"): {"measurement_condition": "free_living"},
            ("COPD", "002"): {"measurement_condition": "free_living"},
        },
        index_cols=["cohort", "participant_id"],
    )


def test_misoriented_dataset_expands_index_by_orientation() -> None:
    """Test that each base row is repeated once per orientation."""
    ds = MisorientedDataset(
        _base_dataset(),
        orientations=["identity", "pa_normal__rot_pa_pos90"],
    )

    assert ds.index.columns.to_list() == ["cohort", "participant_id", "orientation"]
    assert len(ds) == 4
    assert ds.index["orientation"].to_list() == [
        "identity",
        "pa_normal__rot_pa_pos90",
        "identity",
        "pa_normal__rot_pa_pos90",
    ]


def test_misoriented_dataset_rotates_full_single_sensor_recording() -> None:
    """Test full-recording rotation while preserving sensor-frame output."""
    base = _base_dataset()
    ds = MisorientedDataset(base, orientations=["pa_normal__rot_pa_pos90"])[0]

    rotation = REORIENTATION_ROTATIONS["pa_normal__rot_pa_pos90"]
    expected = to_sensor_frame(flip_dataset(to_body_frame(base[0].data_ss), rotation))

    assert ds.orientation_label == "pa_normal__rot_pa_pos90"
    assert_frame_equal(ds.data_ss[SF_SENSOR_COLS], expected)
    assert_frame_equal(base[0].data_ss, _base_dataset()[0].data_ss)


def test_misoriented_dataset_delegates_metadata_and_reference_parameters() -> None:
    """Test metadata and reference access delegates to the wrapped dataset."""
    base = _base_dataset()
    ds = MisorientedDataset(base, orientations=["identity"])[0]

    assert ds.sampling_rate_hz == base[0].sampling_rate_hz
    assert ds.participant_metadata == base[0].participant_metadata
    assert ds.recording_metadata == base[0].recording_metadata
    assert_frame_equal(ds.reference_parameters_.wb_list, base[0].reference_parameters_.wb_list)
