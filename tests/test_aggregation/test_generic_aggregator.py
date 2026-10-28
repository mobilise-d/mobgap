import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.aggregation import GenericAggregator


@pytest.fixture
def example_sdmo_data():
    return pd.DataFrame(
        {
            "visit_type": ["T1"] * 4,
            "participant_id": ["P001", "P001", "P002", "P002"],
            "measurement_date": ["2026-01-01"] * 4,
            "wb_id": [0, 1, 0, 1],
            "duration_s": [12.0, 20.0, 35.0, 75.0],
            "rms_acc_is": [1.0, 3.0, 2.0, 6.0],
            "cv_stride_duration_s": [5.0, 15.0, 10.0, 30.0],
        }
    ).set_index(["visit_type", "participant_id", "measurement_date", "wb_id"])


@pytest.fixture
def dummy_dmo_data_mask(example_sdmo_data):
    return pd.DataFrame(
        np.full(example_sdmo_data.shape, True), index=example_sdmo_data.index, columns=example_sdmo_data.columns
    )


class TestMetaGenericAggregator(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = GenericAggregator

    @pytest.fixture
    def after_action_instance(self, example_sdmo_data):
        return self.ALGORITHM_CLASS(**self.ALGORITHM_CLASS.PredefinedParameters.cvs_sdmo_data).aggregate(
            example_sdmo_data
        )


class TestGenericAggregator:
    def test_default_statistics_are_calculated_separately_per_group(self, example_sdmo_data):
        result = GenericAggregator(**GenericAggregator.PredefinedParameters.cvs_sdmo_data).aggregate(example_sdmo_data)
        expected = pd.DataFrame(
            {
                "all__rms_acc_is__median": [2.0, 4.0],
                "all__rms_acc_is__std": [np.sqrt(2), np.sqrt(8)],
                "all__rms_acc_is__p10": [1.2, 2.4],
                "all__rms_acc_is__p90": [2.8, 5.6],
            },
            index=example_sdmo_data.index.droplevel("wb_id").unique(),
        )
        assert_frame_equal(result.aggregated_data_[expected.columns], expected)

    def test_mask_is_applied_per_measure(self):
        wb_dmos = pd.DataFrame(
            {"duration_s": [12.0, 20.0], "x": [1.0, 3.0], "y": [10.0, 20.0]},
            index=pd.Index([0, 1], name="wb_id"),
        )
        wb_dmos_mask = pd.DataFrame(True, index=wb_dmos.index, columns=wb_dmos.columns)
        wb_dmos_mask.loc[0, "x"] = False
        original_data = wb_dmos.copy(deep=True)
        original_mask = wb_dmos_mask.copy(deep=True)

        result = GenericAggregator(
            groupby=None,
            duration_filters={"all": (0, np.inf)},
            metrics={"mean": "mean"},
            unique_wb_id_column="wb_id",
        ).aggregate(wb_dmos, wb_dmos_mask=wb_dmos_mask)

        assert result.aggregated_data_.loc["all_wbs", "all__x__mean"] == 3.0
        assert result.aggregated_data_.loc["all_wbs", "all__y__mean"] == 15.0
        assert_frame_equal(wb_dmos, original_data)
        assert_frame_equal(wb_dmos_mask, original_mask)

    def test_wb_id_can_be_a_column(self):
        wb_dmos = pd.DataFrame({"wb_id": [0, 1], "duration_s": [12.0, 20.0], "x": [1.0, 3.0]})

        result = GenericAggregator(
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

        result = GenericAggregator(
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
            GenericAggregator(
                **GenericAggregator.PredefinedParameters.cvs_sdmo_data,
            ).aggregate(pd.DataFrame({"duration_s": [12.0], "rms_acc_is": [1.0]}))

    def test_raise_error_on_wrong_groupby(self, example_sdmo_data):
        with pytest.raises(ValueError):
            GenericAggregator(
                **{**GenericAggregator.PredefinedParameters.cvs_sdmo_data, "groupby": ["do", "not", "exist"]}
            ).aggregate(example_sdmo_data)

    def test_raise_error_on_wrong_data_mask(self, example_sdmo_data, dummy_dmo_data_mask):
        with pytest.raises(ValueError):
            GenericAggregator(
                **GenericAggregator.PredefinedParameters.cvs_sdmo_data,
            ).aggregate(example_sdmo_data, wb_dmos_mask=dummy_dmo_data_mask.iloc[:-1])

    def test_raise_warning_on_missing_duration_column(self, example_sdmo_data):
        with pytest.warns(UserWarning):
            GenericAggregator(
                **GenericAggregator.PredefinedParameters.cvs_sdmo_data,
            ).aggregate(example_sdmo_data.drop(columns=["duration_s"]))

    def test_nan_considered_true(self, example_sdmo_data, dummy_dmo_data_mask):
        data = example_sdmo_data.copy()
        data_mask_wit_nan = dummy_dmo_data_mask.copy().replace(True, np.nan)

        agg_with_nan = GenericAggregator(
            **GenericAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_dmos_mask=data_mask_wit_nan)
        agg_without_nan = GenericAggregator(
            **GenericAggregator.PredefinedParameters.cvs_sdmo_data,
        ).aggregate(data, wb_dmos_mask=dummy_dmo_data_mask)

        assert_frame_equal(agg_with_nan.aggregated_data_, agg_without_nan.aggregated_data_)
