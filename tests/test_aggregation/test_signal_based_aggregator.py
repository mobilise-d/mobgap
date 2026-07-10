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
        np.full(example_sdmo_data.shape, True), index=example_sdmo_data.index, columns=example_sdmo_data.columns
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
    def test_mask_is_applied_per_measure(self):
        wb_dmos = pd.DataFrame(
            {"duration_s": [12.0, 20.0], "x": [1.0, 3.0], "y": [10.0, 20.0]},
            index=pd.Index([0, 1], name="wb_id"),
        )
        wb_dmos_mask = pd.DataFrame(True, index=wb_dmos.index, columns=wb_dmos.columns)
        wb_dmos_mask.loc[0, "x"] = False

        result = SDMOAggregator(
            groupby=None,
            duration_filters={"all": (0, np.inf)},
            metrics={"mean": "mean"},
            unique_wb_id_column="wb_id",
        ).aggregate(wb_dmos, wb_dmos_mask=wb_dmos_mask)

        assert result.aggregated_data_.loc["all_wbs", "all__x__mean"] == 3.0
        assert result.aggregated_data_.loc["all_wbs", "all__y__mean"] == 15.0

    def test_wb_id_can_be_a_column(self):
        wb_dmos = pd.DataFrame({"wb_id": [0, 1], "duration_s": [12.0, 20.0], "x": [1.0, 3.0]})

        result = SDMOAggregator(
            groupby=None,
            duration_filters={"all": (0, np.inf)},
            metrics={"mean": "mean"},
            unique_wb_id_column="wb_id",
        ).aggregate(wb_dmos)

        assert result.aggregated_data_.loc["all_wbs", "all__x__mean"] == 2.0
        assert all("wb_id" not in column for column in result.aggregated_data_.columns)

    def test_duration_filters_follow_mobilised_boundaries(self):
        wb_dmos = pd.DataFrame(
            {"duration_s": [10.0, 30.0, 60.0, 61.0], "x": [10.0, 30.0, 60.0, 61.0]},
            index=pd.Index(range(4), name="wb_id"),
        )

        result = SDMOAggregator(
            groupby=None,
            duration_filters={"wb_10_30": (10, 30), "wb_30": (30, np.inf), "wb_60": (60, np.inf)},
            metrics={"mean": "mean"},
            unique_wb_id_column="wb_id",
        ).aggregate(wb_dmos)

        row = result.aggregated_data_.loc["all_wbs"]
        assert row["wb_10_30__x__mean"] == 30.0
        assert row["wb_30__x__mean"] == 60.5
        assert row["wb_60__x__mean"] == 61.0

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
            ).aggregate(example_sdmo_data, wb_dmos_mask=dummy_dmo_data_mask.iloc[:10])

    def test_raise_warning_on_missing_duration_column(self, example_sdmo_data):
        with pytest.warns(UserWarning):
            SDMOAggregator(
                **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
            ).aggregate(example_sdmo_data.drop(columns=["duration_s"]))

    def test_input_not_modified(self, example_sdmo_data, dummy_dmo_data_mask):
        data = example_sdmo_data.copy()
        agg = SDMOAggregator(
            **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_dmos_mask=dummy_dmo_data_mask)
        # check that no rows were dropped
        assert data.shape == agg.filtered_wb_dmos_.shape
        # check that input data is still the same
        assert_frame_equal(data, agg.wb_dmos)
        assert_frame_equal(dummy_dmo_data_mask, agg.wb_dmos_mask)

    def test_nan_considered_true(self, example_sdmo_data, dummy_dmo_data_mask):
        data = example_sdmo_data.copy()
        data_mask_wit_nan = dummy_dmo_data_mask.copy().replace(True, np.nan)

        agg_with_nan = SDMOAggregator(
            **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_dmos_mask=data_mask_wit_nan)
        agg_without_nan = SDMOAggregator(
            **SDMOAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_dmos_mask=dummy_dmo_data_mask)

        assert_frame_equal(agg_with_nan.aggregated_data_, agg_without_nan.aggregated_data_)

    def test_no_grouping(self, example_sdmo_data, dummy_dmo_data_mask):
        data = example_sdmo_data.copy()
        agg = SDMOAggregator(**(SDMOAggregator.PredefinedParameters.cvs_sdmo_data | dict(groupby=None))).aggregate(
            data, wb_dmos_mask=dummy_dmo_data_mask
        )

        assert len(agg.aggregated_data_) == 1
        assert agg.aggregated_data_.index[0] == "all_wbs"
