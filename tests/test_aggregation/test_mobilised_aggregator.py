import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap import PROJECT_ROOT
from mobgap.aggregation import MobilisedAggregator

DATA_PATH = PROJECT_ROOT / "example_data/original_results/mobilised_aggregator"


@pytest.fixture
def example_dmo_data():
    return (
        pd.read_csv(DATA_PATH / "aggregation_test_input.csv")
        .astype({"measurement_date": "string", "visit_type": "string", "participant_id": "string"})
        .set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])
    )


@pytest.fixture
def example_dmo_reference():
    return (
        pd.read_csv(DATA_PATH / "aggregation_test_reference.csv")
        .astype({"measurement_date": "string", "visit_type": "string", "participant_id": "string"})
        .set_index(["visit_type", "participant_id", "measurement_date"])
    )


@pytest.fixture
def dummy_dmo_data_mask(example_dmo_data):
    return example_dmo_data.astype(bool)


@pytest.fixture
def example_dmo_data_partial(example_dmo_data):
    drop_columns = ["n_steps", "n_turns"]
    return example_dmo_data.drop(columns=drop_columns)


@pytest.fixture
def example_dmo_reference_partial(example_dmo_reference):
    drop_columns = ["wbsteps_all_sum", "turns_all_sum"]
    return example_dmo_reference.drop(columns=drop_columns)


class TestMetaMobilisedAggregator(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedAggregator

    @pytest.fixture
    def after_action_instance(self, example_dmo_data):
        return self.ALGORITHM_CLASS().aggregate(
            example_dmo_data.iloc[:10],
            wb_dmos_mask=None,
        )


class TestMobilisedAggregator:
    """Tests for MobilisedAggregator."""

    # columns that show deviations from reference data because of differences in quantile calculation.
    # Walkdur was also rounded differently in the reference data, so we also include it here.
    uncertain_columns = ["ws_30_p90", "cadence_30_p90", "walkdur_all_sum"]

    @pytest.mark.parametrize(
        ("data", "reference"),
        [("example_dmo_data", "example_dmo_reference"), ("example_dmo_data_partial", "example_dmo_reference_partial")],
    )
    def test_reference_data(self, data, reference, request):
        data = request.getfixturevalue(data).rename(columns={"n_steps": "n_raw_initial_contacts"})
        reference = request.getfixturevalue(reference).sort_index(axis=1)
        reference["walkdur_all_sum"] *= 60  # 10.7.2025: we decided to change the reference data to use minutes instead
        # of hours

        agg = MobilisedAggregator(**MobilisedAggregator.PredefinedParameters.cvs_dmo_data).aggregate(data)
        output = agg.aggregated_data_.sort_index(axis=1)

        # For compatibility, we transform the stride length to cm
        for c in ["strlen_1030_avg", "strlen_30_avg"]:
            output[c] *= 100

        # Further we reintroduce rounding:
        output = output.round(3)

        assert_frame_equal(
            output.drop(columns=self.uncertain_columns),
            reference.drop(columns=self.uncertain_columns),
            check_dtype=False,
        )
        assert_frame_equal(output[self.uncertain_columns], reference[self.uncertain_columns], atol=0.05)

    def test_reference_data_with_duration_mask(self, example_dmo_data, dummy_dmo_data_mask, example_dmo_reference):
        dummy_dmo_data_mask = dummy_dmo_data_mask.copy()
        # If all durations are false, all data should be dropped
        dummy_dmo_data_mask.loc[:, "duration_s"] = False
        agg = MobilisedAggregator(
            **MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
        ).aggregate(example_dmo_data, wb_dmos_mask=dummy_dmo_data_mask)
        assert (agg.aggregated_data_["wb_all_sum"] == 0).all()
        # Check for some columns that they are all none
        for col in ["wbdur_all_avg", "wbdur_all_var", "strdur_30_avg", "ws_30_var"]:
            assert agg.aggregated_data_[col].isna().all()

    def test_raise_error_on_wrong_data(self):
        with pytest.raises(ValueError):
            MobilisedAggregator(
                **MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
            ).aggregate(pd.DataFrame(np.random.rand(10, 10)))

    def test_raise_error_on_wrong_groupby(self, example_dmo_data):
        with pytest.raises(ValueError):
            MobilisedAggregator(
                **{**MobilisedAggregator.PredefinedParameters.cvs_dmo_data, "groupby": ["do", "not", "exist"]}
            ).aggregate(example_dmo_data)

    def test_raise_error_on_wrong_data_mask(self, example_dmo_data, dummy_dmo_data_mask):
        with pytest.raises(ValueError):
            MobilisedAggregator(
                **MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
            ).aggregate(example_dmo_data, wb_dmos_mask=dummy_dmo_data_mask.iloc[:10])

    def test_raise_warning_on_missing_duration_column(self, example_dmo_data):
        with pytest.warns(UserWarning):
            MobilisedAggregator(
                **MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
            ).aggregate(example_dmo_data.drop(columns=["duration_s"]))

    def test_input_not_modified(self, example_dmo_data, dummy_dmo_data_mask):
        data = example_dmo_data.copy()
        data_mask = dummy_dmo_data_mask.copy()
        agg = MobilisedAggregator(
            **MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
        ).aggregate(data, wb_dmos_mask=data_mask)
        # check that no rows were dropped
        assert data.shape == agg.filtered_wb_dmos_.shape
        # check that input data is still the same
        assert_frame_equal(data, agg.wb_dmos)
        assert_frame_equal(data_mask, agg.wb_dmos_mask)

    def test_nan_considered_true(self, example_dmo_data, dummy_dmo_data_mask):
        data = example_dmo_data.copy()
        data_mask = dummy_dmo_data_mask.copy()
        with pd.option_context("future.no_silent_downcasting", True):
            data_mask_wit_nan = data_mask.copy().replace(True, np.nan).infer_objects(copy=False)

        agg_with_nan = MobilisedAggregator(
            **MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
        ).aggregate(data, wb_dmos_mask=data_mask_wit_nan)
        agg_without_nan = MobilisedAggregator(
            **MobilisedAggregator.PredefinedParameters.cvs_dmo_data,
        ).aggregate(data, wb_dmos_mask=data_mask)

        assert_frame_equal(agg_with_nan.aggregated_data_, agg_without_nan.aggregated_data_)

    def test_no_grouping(self, example_dmo_data, dummy_dmo_data_mask):
        data = example_dmo_data.copy()
        data_mask = dummy_dmo_data_mask.copy()
        agg = MobilisedAggregator(
            **(MobilisedAggregator.PredefinedParameters.cvs_dmo_data | dict(groupby=None))
        ).aggregate(data, wb_dmos_mask=data_mask)

        assert len(agg.aggregated_data_) == 1
        assert agg.aggregated_data_.index[0] == "all_wbs"

    def test_alternative_names(self, example_dmo_data):
        data = example_dmo_data.copy()
        # With alternative names
        agg = MobilisedAggregator().aggregate(data)
        # Without alternative names
        agg_original = MobilisedAggregator(**MobilisedAggregator.PredefinedParameters.cvs_dmo_data).aggregate(data)

        assert len(set(agg.aggregated_data_.columns).intersection(set(agg_original.aggregated_data_.columns))) == 0
        assert len(agg_original.aggregated_data_.columns) == len(agg.aggregated_data_.columns)
