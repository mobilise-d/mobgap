import pandas as pd

from mobgap.consts import BF_SENSOR_COLS, SF_SENSOR_COLS
from mobgap.data import GaitDatasetFromData
from mobgap.data.base import ReferenceData
from mobgap.re_orientation.evaluation import MisorientedDataset


class ReferenceGaitDatasetFromData(GaitDatasetFromData):
    def __init__(
        self,
        _data,
        _sampling_rate_hz,
        _participant_metadata=None,
        _recording_metadata=None,
        *,
        single_sensor_name="LowerBack",
        index_cols=None,
        groupby_cols=None,
        subset_index=None,
        reference_parameters=None,
    ) -> None:
        self.reference_parameters = reference_parameters
        super().__init__(
            _data,
            _sampling_rate_hz,
            _participant_metadata,
            _recording_metadata,
            single_sensor_name=single_sensor_name,
            index_cols=index_cols,
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    @property
    def reference_parameters_(self) -> ReferenceData:
        return self.reference_parameters

    @property
    def reference_parameters_relative_to_wb_(self) -> ReferenceData:
        return self.reference_parameters

    @property
    def reference_sampling_rate_hz_(self) -> float:
        return self.sampling_rate_hz


def _minimal_reference() -> ReferenceData:
    return ReferenceData(
        wb_list=pd.DataFrame({"start": [0], "end": [3]}),
        ic_list=None,
        turn_parameters=None,
        stride_parameters=None,
    )


def test_misoriented_dataset_extends_index_with_orientation() -> None:
    base_dataset = ReferenceGaitDatasetFromData(
        {("p1", "r1"): {"LowerBack": pd.DataFrame(index=range(3))}},
        100.0,
        {("p1", "r1"): {"cohort": "HA", "height_m": 1.7, "sensor_height_m": 1.0}},
        {("p1", "r1"): {"measurement_condition": "laboratory"}},
        index_cols=["participant_id", "recording"],
        reference_parameters=_minimal_reference(),
    )

    dataset = MisorientedDataset(
        base_dataset,
        orientations=["identity", "pa_normal__rot_pa_180"],
    )

    assert list(dataset.index.columns) == ["participant_id", "recording", "orientation"]
    assert dataset.index.to_dict("records") == [
        {"participant_id": "p1", "recording": "r1", "orientation": "identity"},
        {"participant_id": "p1", "recording": "r1", "orientation": "pa_normal__rot_pa_180"},
    ]

    datapoint = dataset.get_subset(participant_id="p1", recording="r1", orientation="identity")
    pd.testing.assert_frame_equal(
        datapoint.reference_parameters_.wb_list,
        base_dataset.reference_parameters_.wb_list,
    )
    assert datapoint.reference_sampling_rate_hz_ == 100.0


def test_misoriented_dataset_body_output_rotates_body_frame_data() -> None:
    body_frame_data = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        columns=BF_SENSOR_COLS,
    )
    base_dataset = ReferenceGaitDatasetFromData(
        {"p1": {"LowerBack": body_frame_data}},
        100.0,
        {"p1": {"cohort": "HA", "height_m": 1.7, "sensor_height_m": 1.0}},
        {"p1": {"measurement_condition": "laboratory"}},
        index_cols="participant_id",
        reference_parameters=_minimal_reference(),
    )

    datapoint = MisorientedDataset(
        base_dataset,
        orientations=["pa_normal__rot_pa_180"],
        output_frame="body",
    )[0]

    expected = pd.DataFrame(
        [[-1.0, -2.0, 3.0, -4.0, -5.0, 6.0]],
        columns=BF_SENSOR_COLS,
    )
    pd.testing.assert_frame_equal(datapoint.data_ss, expected)


def test_misoriented_dataset_body_output_converts_sensor_frame_data_before_rotation() -> None:
    sensor_frame_data = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        columns=SF_SENSOR_COLS,
    )
    base_dataset = ReferenceGaitDatasetFromData(
        {"p1": {"LowerBack": sensor_frame_data}},
        100.0,
        {"p1": {"cohort": "HA", "height_m": 1.7, "sensor_height_m": 1.0}},
        {"p1": {"measurement_condition": "laboratory"}},
        index_cols="participant_id",
        reference_parameters=_minimal_reference(),
    )

    datapoint = MisorientedDataset(base_dataset, orientations=["identity"], output_frame="body")[0]

    expected = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        columns=BF_SENSOR_COLS,
    )
    pd.testing.assert_frame_equal(datapoint.data_ss, expected)


def test_misoriented_dataset_same_output_preserves_sensor_frame_data() -> None:
    sensor_frame_data = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        columns=SF_SENSOR_COLS,
    )
    base_dataset = ReferenceGaitDatasetFromData(
        {"p1": {"LowerBack": sensor_frame_data}},
        100.0,
        {"p1": {"cohort": "HA", "height_m": 1.7, "sensor_height_m": 1.0}},
        {"p1": {"measurement_condition": "laboratory"}},
        index_cols="participant_id",
        reference_parameters=_minimal_reference(),
    )

    datapoint = MisorientedDataset(
        base_dataset,
        orientations=["pa_normal__rot_pa_180"],
        output_frame="same",
    )[0]

    expected = pd.DataFrame(
        [[-1.0, -2.0, 3.0, -4.0, -5.0, 6.0]],
        columns=SF_SENSOR_COLS,
    )
    pd.testing.assert_frame_equal(datapoint.data_ss, expected)
