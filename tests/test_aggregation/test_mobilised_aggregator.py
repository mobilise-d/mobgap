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
    return pd.read_csv(DATA_PATH / "aggregation_test_input.csv", index_col=0)


@pytest.fixture()
def example_dmo_data_mask():
    return pd.read_csv(DATA_PATH / "aggregation_test_data_mask.csv", index_col=0)


@pytest.fixture()
def example_dmo_reference():
    return pd.read_csv(DATA_PATH / "aggregation_test_reference.csv", index_col=[0, 1])


@pytest.fixture()
def example_dmo_data_partial():
    drop_columns = ["step_number", "turn_number"]
    return pd.read_csv(DATA_PATH / "aggregation_test_input.csv", index_col=0).drop(columns=drop_columns)


@pytest.fixture()
def example_dmo_reference_partial():
    drop_columns = ["steps_all_sum", "turns_all_sum"]
    return pd.read_csv(DATA_PATH / "aggregation_test_reference.csv", index_col=[0, 1]).drop(columns=drop_columns)


class TestMetaMobilisedAggregator(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = MobilisedAggregator

    @pytest.fixture()
    def after_action_instance(self, example_dmo_data):
        return self.ALGORITHM_CLASS().aggregate(
            example_dmo_data.iloc[:10],
            data_mask=None,
        )


class TestMobilisedAggregator:
    """Tests for MobilisedAggregator."""

    def test_reference_data(self, example_dmo_data, example_dmo_data_mask, example_dmo_reference):
        agg = MobilisedAggregator().aggregate(example_dmo_data, data_mask=example_dmo_data_mask)
        output = agg.aggregated_data_
        assert_frame_equal(
            output.drop(columns=["ws_30_max", "cadence_30_max"]),
            example_dmo_reference.drop(columns=["ws_30_max", "cadence_30_max"]),
            check_dtype=False,
        )

    def test_incomplete_reference_data(
        self, example_dmo_data_partial, example_dmo_data_mask, example_dmo_reference_partial
    ):
        agg = MobilisedAggregator().aggregate(example_dmo_data_partial, data_mask=example_dmo_data_mask)
        output = agg.aggregated_data_
        assert_frame_equal(
            output.drop(columns=["ws_30_max", "cadence_30_max"]),
            example_dmo_reference_partial.drop(columns=["ws_30_max", "cadence_30_max"]),
            check_dtype=False,
        )

    def test_reference_data_with_duration_mask(self, example_dmo_data, example_dmo_data_mask, example_dmo_reference):
        example_dmo_data_mask["duration"] = [True] * len(example_dmo_data_mask)
        agg = MobilisedAggregator().aggregate(
            example_dmo_data, data_mask=example_dmo_data_mask, duration_mask=example_dmo_data_mask
        )
        output = agg.aggregated_data_
        assert_frame_equal(
            output.drop(columns=["ws_30_max", "cadence_30_max"]),
            example_dmo_reference.drop(columns=["ws_30_max", "cadence_30_max"]),
            check_dtype=False,
        )

    def test_raise_error_on_wrong_data(self):
        with pytest.raises(ValueError):
            MobilisedAggregator().aggregate(pd.DataFrame(np.random.rand(10, 10)))

    def test_raise_error_on_wrong_groupby(self, example_dmo_data):
        with pytest.raises(ValueError):
            MobilisedAggregator(groupby_columns=["do", "not", "exist"]).aggregate(example_dmo_data)

    def test_raise_error_on_wrong_data_mask(self, example_dmo_data, example_dmo_data_mask):
        with pytest.raises(ValueError):
            MobilisedAggregator().aggregate(example_dmo_data, data_mask=example_dmo_data_mask.iloc[:10])

    def test_raise_warning_on_missing_duration_column(self, example_dmo_data):
        with pytest.warns(UserWarning):
            MobilisedAggregator().aggregate(example_dmo_data.drop(columns=["duration"]))

    def test_input_not_modified(self, example_dmo_data, example_dmo_data_mask):
        data = example_dmo_data.copy()
        data_mask = example_dmo_data_mask.copy()
        agg = MobilisedAggregator().aggregate(data, data_mask=data_mask)
        # check that no rows were dropped
        assert data.shape == agg.filtered_data_.shape
        # check that input data is still the same
        assert_frame_equal(data, agg.data)
        assert_frame_equal(data_mask, agg.data_mask)
