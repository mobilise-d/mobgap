from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from gaitlink.aggregation import MobilisedAggregator

BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / "example_data/original_results/mobilised_aggregator"


@pytest.fixture()
def example_dmo_data():
    return (
        pd.read_csv(DATA_PATH / "aggregation_test_input.csv")
        .astype({"measurement_date": "string", "visit_type": "string", "participant_id": "string"})
        .set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])
    )


@pytest.fixture()
def example_dmo_reference():
    return (
        pd.read_csv(DATA_PATH / "aggregation_test_reference.csv")
        .astype({"measurement_date": "string", "visit_type": "string", "participant_id": "string"})
        .set_index(["visit_type", "participant_id", "measurement_date"])
    )


@pytest.fixture()
def dummy_dmo_data_mask(example_dmo_data):
    return example_dmo_data.astype(bool)


@pytest.fixture()
def example_dmo_data_partial(example_dmo_data):
    drop_columns = ["n_steps", "n_turns"]
    return example_dmo_data.drop(columns=drop_columns)


@pytest.fixture()
def example_dmo_reference_partial(example_dmo_reference):
    drop_columns = ["steps_all_sum", "turns_all_sum"]
    return example_dmo_reference.drop(columns=drop_columns)


class TestMetaMobilisedAggregator(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedAggregator

    @pytest.fixture()
    def after_action_instance(self, example_dmo_data):
        return self.ALGORITHM_CLASS().aggregate(
            example_dmo_data.iloc[:10],
            wb_dmos_mask=None,
        )


class TestMobilisedAggregator:
    """Tests for MobilisedAggregator."""

    # columns that show deviations from reference data because of differences in quantile calculation
    quantile_columns = ["ws_30_max", "cadence_30_max"]

    @pytest.mark.parametrize(
        ("data", "reference"),
        [("example_dmo_data", "example_dmo_reference"), ("example_dmo_data_partial", "example_dmo_reference_partial")],
    )
    def test_reference_data(self, data, reference, request):
        data = request.getfixturevalue(data)
        reference = request.getfixturevalue(reference).sort_index(axis=1)

        agg = MobilisedAggregator().aggregate(data)
        output = agg.aggregated_data_.sort_index(axis=1)

        assert_frame_equal(
            output.drop(columns=self.quantile_columns),
            reference.drop(columns=self.quantile_columns),
            check_dtype=False,
        )
        assert_frame_equal(output[self.quantile_columns], reference[self.quantile_columns], atol=0.05)

    def test_reference_data_with_duration_mask(self, example_dmo_data, dummy_dmo_data_mask, example_dmo_reference):
        dummy_dmo_data_mask = dummy_dmo_data_mask.copy()
        # If all durations are false, all data should be dropped
        dummy_dmo_data_mask.loc[:, "duration_s"] = False
        agg = MobilisedAggregator().aggregate(example_dmo_data, wb_dmos_mask=dummy_dmo_data_mask)
        assert (agg.aggregated_data_["wb_all_sum"] == 0).all()
        # Check for some columns that they are all none
        for col in ["wbdur_all_avg", "wbdur_all_var", "strdur_30_avg", "ws_30_var"]:
            assert agg.aggregated_data_[col].isna().all()

    def test_raise_error_on_wrong_data(self):
        with pytest.raises(ValueError):
            MobilisedAggregator().aggregate(pd.DataFrame(np.random.rand(10, 10)))

    def test_raise_error_on_wrong_groupby(self, example_dmo_data):
        with pytest.raises(ValueError):
            MobilisedAggregator(groupby=["do", "not", "exist"]).aggregate(example_dmo_data)

    def test_raise_error_on_wrong_data_mask(self, example_dmo_data, dummy_dmo_data_mask):
        with pytest.raises(ValueError):
            MobilisedAggregator().aggregate(example_dmo_data, wb_dmos_mask=dummy_dmo_data_mask.iloc[:10])

    def test_raise_warning_on_missing_duration_column(self, example_dmo_data):
        with pytest.warns(UserWarning):
            MobilisedAggregator().aggregate(example_dmo_data.drop(columns=["duration_s"]))

    def test_input_not_modified(self, example_dmo_data, dummy_dmo_data_mask):
        data = example_dmo_data.copy()
        data_mask = dummy_dmo_data_mask.copy()
        agg = MobilisedAggregator().aggregate(data, wb_dmos_mask=data_mask)
        # check that no rows were dropped
        assert data.shape == agg.filtered_wb_dmos_.shape
        # check that input data is still the same
        assert_frame_equal(data, agg.wb_dmos)
        assert_frame_equal(data_mask, agg.wb_dmos_mask)

    def test_nan_considered_true(self, example_dmo_data, dummy_dmo_data_mask):
        data = example_dmo_data.copy()
        data_mask = dummy_dmo_data_mask.copy()
        data_mask_wit_nan = data_mask.copy().replace(True, np.nan)

        agg_with_nan = MobilisedAggregator().aggregate(data, wb_dmos_mask=data_mask_wit_nan)
        agg_without_nan = MobilisedAggregator().aggregate(data, wb_dmos_mask=data_mask)

        assert_frame_equal(agg_with_nan.aggregated_data_, agg_without_nan.aggregated_data_)
