"""Tests for WtdMegaritisXGBoost algorithm."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
from mobgap.weartime import WtdMegaritisXGBoost


class TestMetaWtdMegaritisXGBoost(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = WtdMegaritisXGBoost

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS), sampling_rate_hz=100.0
        )


class TestWtdMegaritisXGBoost:
    """Tests for WtdMegaritisXGBoost.

    Note: We don't test the influence of any single parameter here.
    We just test the happy path and some potential edge cases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_no_weartime(self):
        """Zero signal should result in no wear-time."""
        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
        output = WtdMegaritisXGBoost().detect(data, sampling_rate_hz=100.0).weartime_list_

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end", "wt_id"]).astype("int64").set_index("wt_id"))

    def test_single_weartime_period(self):
        """Test detection of single wear-time period."""
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test11", trial="Trial1").data_ss

        output = WtdMegaritisXGBoost().detect(to_body_frame(data), sampling_rate_hz=100.0).weartime_list_

        assert len(output) >= 1
        assert set(output.columns) == {"start", "end"}

    @pytest.mark.parametrize("version", ["full", "lightweight"])
    def test_both_model_versions(self, version):
        """Test that both model versions work correctly."""
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = WtdMegaritisXGBoost(version=version).detect(to_body_frame(data), sampling_rate_hz=100.0)

        # Should complete successfully and produce valid output
        assert hasattr(output, "weartime_list_")
        assert hasattr(output, "model")
        assert set(output.weartime_list_.columns) == {"start", "end"}

    def test_waking_hours_attribute_exists(self):
        """Verify total_weartime_hours_during_waking_ is calculated."""
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = WtdMegaritisXGBoost().detect(to_body_frame(data), sampling_rate_hz=100.0)

        assert hasattr(output, "total_weartime_hours_during_waking_")
        assert output.total_weartime_hours_during_waking_ >= 0


class TestWtdMegaritisXGBoostRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    @pytest.mark.parametrize("version", ["full", "lightweight"])
    def test_example_lab_data(self, datapoint, version, snapshot):
        """Test on all LabExampleDataset datapoints with both model versions."""
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        weartime_list = (
            WtdMegaritisXGBoost(version=version)
            .detect(to_body_frame(data), sampling_rate_hz=sampling_rate_hz)
            .weartime_list_
        )

        # Include version in snapshot name to separate full vs lightweight results
        snapshot.assert_match(weartime_list, f"{version}_{tuple(datapoint.group_label)!s}")
