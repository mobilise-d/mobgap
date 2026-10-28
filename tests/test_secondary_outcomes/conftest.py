import pytest

from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame


@pytest.fixture
def example_walking_bout():
    data = LabExampleDataset(reference_system="INDIP").get_subset(test="Test11")[0]
    wb_id = data.reference_parameters_relative_to_wb_.turn_parameters.index.get_level_values(0)[0]
    turn_list = data.reference_parameters_relative_to_wb_.turn_parameters.loc[wb_id]
    stride_list = data.reference_parameters_relative_to_wb_.stride_parameters.loc[wb_id].rename(
        columns={"duration_s": "stride_duration_s", "length_m": "stride_length_m"}
    )
    stride_list["cadence_spm"] = 120 * stride_list["speed_mps"] / stride_list["stride_length_m"]
    sampling_rate_hz = data.sampling_rate_hz
    data_start = int(data.reference_parameters_.wb_list.loc[wb_id, "start"])
    data_end = int(data.reference_parameters_.wb_list.loc[wb_id, "end"])
    data = to_body_frame(data.data_ss.iloc[data_start:data_end])
    return data, stride_list, turn_list, sampling_rate_hz
