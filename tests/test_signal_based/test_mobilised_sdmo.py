import pandas as pd
import pytest
from tpcp.testing import TestAlgorithmMixin

from mobgap.data import LabExampleDataset
from mobgap.signal_based import (
    HarmonicRatio,
    MobilisedSDMO,
    TurnSDMO,
)
from mobgap.utils.conversions import to_body_frame


@pytest.fixture
def example_walking_bout():
    data = LabExampleDataset(reference_system="INDIP").get_subset(test="Test11")[0]
    wb_id = data.reference_parameters_relative_to_wb_.turn_parameters.index.get_level_values(0)[0]
    turn_list = data.reference_parameters_relative_to_wb_.turn_parameters.loc[wb_id]
    stride_list = data.reference_parameters_relative_to_wb_.stride_parameters.loc[wb_id].rename(
        columns={"duration_s": "stride_duration_s", "length_m": "stride_length_m"}
    )
    stride_list["cadence_spm"] = 60 * stride_list["speed_mps"] / stride_list["stride_length_m"]
    sampling_rate_hz = data.sampling_rate_hz
    data_start = data.reference_parameters_.stride_parameters.loc[wb_id][["start", "end"]].min().min()
    data_end = data.reference_parameters_.stride_parameters.loc[wb_id][["start", "end"]].max().max()
    data = to_body_frame(data.data_ss.iloc[data_start:data_end])
    return data, stride_list, turn_list, sampling_rate_hz


class TestMetaMobilisedSDMO(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = MobilisedSDMO
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self, example_walking_bout):
        data, stride_list, turn_list, sampling_rate_hz = example_walking_bout
        return self.ALGORITHM_CLASS(**self.ALGORITHM_CLASS.PredefinedParameters.default).calculate(
            data=data,
            sampling_rate_hz=100.0,
            stride_list=stride_list,
            turn_list=turn_list,
            replicate_matlab=True,
        )


class TestMobilisedSDMORegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP"))
    @pytest.mark.parametrize("replicate_matlab", [True, False])
    def test_regression(self, datapoint, replicate_matlab, snapshot):
        stride_meta = datapoint.reference_parameters_relative_to_wb_.stride_parameters
        if stride_meta.empty:
            pytest.skip("No stride parameters available for this datapoint")
        wb_ids = stride_meta.index.get_level_values(0).unique()

        for wb_id in wb_ids:
            stride_list = stride_meta.loc[wb_id].rename(
                columns={"duration_s": "stride_duration_s", "length_m": "stride_length_m"}
            )
            if stride_list.empty:
                continue
            stride_list["cadence_spm"] = 60 * stride_list["speed_mps"] / stride_list["stride_length_m"]
            turn_meta = datapoint.reference_parameters_relative_to_wb_.turn_parameters
            if not turn_meta.empty and wb_id in turn_meta.index.get_level_values(0):
                turn_list = turn_meta.loc[wb_id]
            else:
                turn_list = pd.DataFrame([])
            sampling_rate_hz = datapoint.sampling_rate_hz
            data_start = stride_list["start"].min()
            data_end = stride_list["end"].max()
            imu_data = to_body_frame(datapoint.data_ss.iloc[data_start:data_end])

            result = MobilisedSDMO().calculate(
                data=imu_data,
                sampling_rate_hz=sampling_rate_hz,
                stride_list=stride_list,
                turn_list=turn_list,
                replicate_matlab=replicate_matlab,
            )
            snapshot_name = f"{tuple(datapoint.group_label)!s}_wb{wb_id}_replicate_{replicate_matlab}"
            snapshot.assert_match(result.signal_based_parameters_, name=snapshot_name)


class TestMobilisedSDMO:
    def test_empty_calculators(self, example_walking_bout):
        result = MobilisedSDMO(calculators=()).calculate(
            data=None,
            sampling_rate_hz=None,
            stride_list=None,
            turn_list=None,
            replicate_matlab=True,
        )
        assert result.signal_based_parameters_.empty

    def test_empty_turn_list(self, example_walking_bout):
        data, stride_list, _, sampling_rate_hz = example_walking_bout
        result = MobilisedSDMO(calculators=(("turn", TurnSDMO()),)).calculate(
            data=data,
            sampling_rate_hz=100.0,
            stride_list=stride_list,
            turn_list=pd.DataFrame([]),
            replicate_matlab=True,
        )
        assert result.signal_based_parameters_.empty

    def test_empty_stride_list(self, example_walking_bout):
        data, _, turn_list, sampling_rate_hz = example_walking_bout
        result = MobilisedSDMO(calculators=(("harmonic_ratio", HarmonicRatio(acc_columns=["acc_is"])),)).calculate(
            data=data,
            sampling_rate_hz=100.0,
            stride_list=pd.DataFrame([]),
            turn_list=turn_list,
            replicate_matlab=True,
        )
        assert result.signal_based_parameters_.empty
