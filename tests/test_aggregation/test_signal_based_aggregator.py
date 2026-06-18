import pandas as pd
import numpy as np
import pytest
from tpcp.testing import TestAlgorithmMixin
from pandas._testing import assert_frame_equal
from mobgap import PROJECT_ROOT
from mobgap.aggregation import SDMOAggregator

DATA_PATH = PROJECT_ROOT / "example_data/original_results/sdmo_aggregator"


@pytest.fixture
def example_sdmo_data():
    return (
        pd.read_csv(DATA_PATH / "sdmo_aggregation_test_input.csv")
        .astype({"measurement_date": "string", "visit_type": "string", "participant_id": "string"})
        .set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])
    )

@pytest.fixture
def dummy_dmo_data_mask(example_sdmo_data):
    return pd.DataFrame(
        np.full(example_sdmo_data.shape, True),
        index=example_sdmo_data.index,
        columns=example_sdmo_data.columns
    )


class TestMetaSDMOAggregator(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = SDMOAggregator

    @pytest.fixture
    def after_action_instance(self, example_sdmo_data):
        return self.ALGORITHM_CLASS().aggregate(
            example_sdmo_data.iloc[:10],
            wb_dmos_mask=None,
        )


class TestSDMOAggregator:
    def test_raise_error_on_wrong_data(self):
        with pytest.raises(ValueError):
            SDMOAggregator(
                **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
            ).aggregate(pd.DataFrame(np.random.rand(10, 10)))

    def test_raise_error_on_wrong_groupby(self, example_sdmo_data):
        with pytest.raises(ValueError):
            SDMOAggregator(
                **{**SDMOAggregator.PredefinedParameters.cvs_sdmo_data, "groupby": ["do", "not", "exist"]}
            ).aggregate(example_sdmo_data)

    def test_raise_error_on_wrong_data_mask(self, example_sdmo_data, dummy_dmo_data_mask):
        with pytest.raises(ValueError):
            SDMOAggregator(
                **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
            ).aggregate(example_sdmo_data, wb_sdmos_mask=dummy_dmo_data_mask.iloc[:10])

    def test_raise_warning_on_missing_duration_column(self, example_sdmo_data):
        with pytest.warns(UserWarning):
            SDMOAggregator(
                **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
            ).aggregate(example_sdmo_data.drop(columns=["duration_s"]))

    def test_input_not_modified(self, example_sdmo_data, dummy_dmo_data_mask):
        data = example_sdmo_data.copy()
        agg = SDMOAggregator(
            **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_sdmos_mask=dummy_dmo_data_mask)
        # check that no rows were dropped
        assert data.shape == agg.filtered_wb_sdmos_.shape
        # check that input data is still the same
        assert_frame_equal(data, agg.wb_sdmos)
        assert_frame_equal(dummy_dmo_data_mask, agg.wb_sdmos_mask)

    def test_nan_considered_true(self, example_sdmo_data, dummy_dmo_data_mask):
        data = example_sdmo_data.copy()
        data_mask_wit_nan = dummy_dmo_data_mask.copy().replace(True, np.nan)

        agg_with_nan = SDMOAggregator(
            **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_sdmos_mask=data_mask_wit_nan)
        agg_without_nan = SDMOAggregator(
            **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_sdmos_mask=dummy_dmo_data_mask)

        assert_frame_equal(agg_with_nan.aggregated_data_, agg_without_nan.aggregated_data_)

    def test_no_grouping(self, example_sdmo_data, dummy_dmo_data_mask):
        data = example_sdmo_data.copy()
        agg = SDMOAggregator(
            **(SDMOAggregator.PredefinedParameters.cvs_sdmo_data | dict(groupby=None))
        ).aggregate(data, wb_dmos_mask=dummy_dmo_data_mask)

        assert len(agg.aggregated_data_) == 1
        assert agg.aggregated_data_.index[0] == "all_wbs"
