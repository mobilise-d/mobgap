"""Tests for WtdMegaritisCNN algorithm."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame
from mobgap.weartime import WtdMegaritisCNN


class TestMetaWtdMegaritisCNN(TestAlgorithmMixin):
    __test__ = True
    ALGORITHM_CLASS = WtdMegaritisCNN

    @pytest.fixture
    def after_action_instance(self):
        # Use random data to avoid edge cases
        return self.ALGORITHM_CLASS().detect(
            pd.DataFrame(np.random.randn(1000, 6), columns=BF_SENSOR_COLS), sampling_rate_hz=100.0
        )


class TestWtdMegaritisCNN:
    """Tests for WtdMegaritisCNN.

    Note: We don't test the influence of any single parameter here.
    We just test the happy path and some potential edge cases.
    If people run into bugs when changing parameters, we can add more tests.
    """

    def test_no_weartime(self):
        """Zero signal should result in no wear-time."""
        data = pd.DataFrame(np.zeros((1000, 6)), columns=BF_SENSOR_COLS)
        output = WtdMegaritisCNN().detect(data, sampling_rate_hz=100.0).weartime_list_

        assert_frame_equal(output, pd.DataFrame(columns=["start", "end", "wt_id"]).astype("int64").set_index("wt_id"))

    def test_single_weartime_period(self):
        """Test detection on a single trial."""
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = WtdMegaritisCNN().detect(to_body_frame(data), sampling_rate_hz=100.0).weartime_list_

        # Verify output structure (model may detect zero wear-time on short trials)
        assert set(output.columns) == {"start", "end"}
        assert output.index.name == "wt_id"

    @pytest.mark.parametrize("version", ["cnn", "cnn_lstm"])
    def test_both_model_versions(self, version):
        """Test that both model architectures work correctly."""
        data = LabExampleDataset().get_subset(cohort="HA", participant_id="001", test="Test5", trial="Trial2").data_ss

        output = WtdMegaritisCNN(version=version).detect(to_body_frame(data), sampling_rate_hz=100.0)

        # Should complete successfully and produce valid output
        assert hasattr(output, "weartime_list_")
        assert hasattr(output, "model")
        assert set(output.weartime_list_.columns) == {"start", "end"}


class TestWtdMegaritisCNNRegression:
    @pytest.mark.parametrize("datapoint", LabExampleDataset(reference_system="INDIP", reference_para_level="wb"))
    @pytest.mark.parametrize("version", ["cnn", "cnn_lstm"])
    def test_example_lab_data(self, datapoint, version, snapshot):
        """Test on all LabExampleDataset datapoints with both model versions."""
        data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        weartime_list = (
            WtdMegaritisCNN(version=version)
            .detect(to_body_frame(data), sampling_rate_hz=sampling_rate_hz)
            .weartime_list_
        )

        # Include version in snapshot name to separate cnn vs cnn_lstm results
        snapshot.assert_match(weartime_list, f"{version}_{tuple(datapoint.group_label)!s}")
