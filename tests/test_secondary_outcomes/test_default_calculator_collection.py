import pandas as pd
import pytest

from mobgap.data import LabExampleDataset
from mobgap.secondary_outcomes import (
    DEFAULT_WB_SDMO_CALCULATORS,
    GaitRegularityAndSymmetry,
    GaitRegularityAndSymmetryTruncatedMean,
)
from mobgap.utils.conversions import to_body_frame


class TestDefaultCalculatorCollectionRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP"))
    @pytest.mark.parametrize(
        "regularity_calculator",
        [
            pytest.param(GaitRegularityAndSymmetry(), id="symmetric_mean"),
            pytest.param(GaitRegularityAndSymmetryTruncatedMean(), id="truncated_mean"),
        ],
    )
    def test_regression(self, datapoint, regularity_calculator, snapshot):
        calculators = dict(DEFAULT_WB_SDMO_CALCULATORS)
        calculators["regularity_symmetry"] = regularity_calculator
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
            stride_list["cadence_spm"] = 120 * stride_list["speed_mps"] / stride_list["stride_length_m"]
            sampling_rate_hz = datapoint.sampling_rate_hz
            data_start = int(datapoint.reference_parameters_.wb_list.loc[wb_id, "start"])
            data_end = int(datapoint.reference_parameters_.wb_list.loc[wb_id, "end"])
            imu_data = to_body_frame(datapoint.data_ss.iloc[data_start:data_end])

            result = pd.concat(
                {
                    name: calculator.clone()
                    .calculate(
                        imu_data,
                        sampling_rate_hz=sampling_rate_hz,
                        stride_list=stride_list,
                    )
                    .signal_based_parameters_
                    for name, calculator in calculators.items()
                },
                names=["algorithm", "parameter"],
            )
            snapshot_name = f"{tuple(datapoint.group_label)!s}_wb{wb_id}"
            snapshot.assert_match(
                result.droplevel("algorithm").to_frame().T.rename_axis(columns=None), name=snapshot_name
            )
