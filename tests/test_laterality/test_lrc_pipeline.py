from collections.abc import Iterable
from typing import Any, Union
from unittest.mock import patch

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin
from typing_extensions import Self, Unpack

from mobgap.data import LabExampleDataset
from mobgap.laterality import LrcMcCamley
from mobgap.laterality.base import BaseLRClassifier
from mobgap.laterality.evaluation import lrc_score
from mobgap.laterality.pipeline import LrcEmulationPipeline


class TestMetaLrcEmulationPipeline(TestAlgorithmMixin):
    __test__ = True

    ALGORITHM_CLASS = LrcEmulationPipeline
    ONLY_DEFAULT_PARAMS = False

    @pytest.fixture
    def after_action_instance(self):
        return self.ALGORITHM_CLASS(LrcMcCamley()).run(
            LabExampleDataset(reference_system="INDIP").get_subset(
                cohort="HA", participant_id="001", test="Test5", trial="Trial2"
            )
        )


class DummyLrc(BaseLRClassifier):
    def __init__(self, ic_lr_list=None):
        self.ic_lr_list = ic_lr_list

    def predict(
        self,
        data: pd.DataFrame,
        ic_list: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ):
        self.ic_lr_list_ = self.ic_lr_list
        return self

    def self_optimize(
        self,
        data_sequences: Iterable[pd.DataFrame],
        ic_list_per_sequence: Iterable[pd.DataFrame],
        ref_ic_lr_list_per_sequence: Iterable[pd.DataFrame],
        *,
        sampling_rate_hz: Union[float, Iterable[float]],
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        self.ic_list = list(ic_list_per_sequence)[0]
        return self


class TestLrcEmulationPipeline:
    def test_simple_run(self):
        data = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test11", trial="Trial1"
        )
        n_wbs = len(data.reference_parameters_.wb_list)
        output = LrcEmulationPipeline(LrcMcCamley()).run(data)
        predictions = output.ic_lr_list_
        assert len(output.per_wb_algo_) == n_wbs
        assert_frame_equal(
            predictions.drop("lr_label", axis=1), data.reference_parameters_.ic_list.drop("lr_label", axis=1)
        )
        assert set(predictions.columns) == {"ic", "lr_label"}

        # We check that the individual algorithms were called with the individual data (we just check for ic_list)
        all_ics = data.reference_parameters_relative_to_wb_.ic_list
        for wb_id, algo in output.per_wb_algo_.items():
            assert_frame_equal(algo.ic_list, all_ics.loc[wb_id].drop("lr_label", axis=1))
            assert algo.__class__ == LrcMcCamley

    def test_simple_self_optimize(self):
        dataset = LabExampleDataset(reference_system="INDIP")[:2]
        dummy_lrc = DummyLrc(None)

        with patch.object(dummy_lrc, "self_optimize", return_value=dummy_lrc) as mock_method:
            pipeline = LrcEmulationPipeline(dummy_lrc)
            pipeline.self_optimize(dataset)

        # You can now assert that the mock was called
        mock_method.assert_called_once()
        call_args = mock_method.call_args_list[0][0]
        call_kwargs = mock_method.call_args_list[0][1]

        # Note: We don't test that the data is passed correctly, checking that would basically require to copy the
        #       entire logic we have.
        assert isinstance(call_args[0], Iterable)
        assert isinstance(call_args[1], Iterable)
        assert isinstance(call_args[2], Iterable)
        assert isinstance(call_kwargs["sampling_rate_hz"], Iterable)
        assert (
            len(list(call_kwargs["sampling_rate_hz"]))
            == len(list(call_args[0]))
            == len(list(call_args[1]))
            == len(list(call_args[2]))
            == sum(len(datapoint.reference_parameters_.wb_list) for datapoint in dataset)
        )

    def test_simple_score(self):
        # We need a datapoint with only a single WB, otherwise our dummy will not work
        datapoint = LabExampleDataset(reference_system="INDIP").get_subset(
            cohort="HA", participant_id="001", test="Test5", trial="Trial1"
        )
        dummy_lrc = DummyLrc(datapoint.reference_parameters_.ic_list.reset_index("wb_id", drop=True))

        pipeline = LrcEmulationPipeline(dummy_lrc)
        agg_scores, single_scores = lrc_score(pipeline, datapoint)

        raw_results = single_scores["raw__predictions"]

        assert agg_scores["accuracy"] == 1.0
        assert isinstance(raw_results, pd.DataFrame)
        assert set(raw_results.columns) == {"reference", "predicted"}
        assert (raw_results["reference"] == raw_results["predicted"]).all()
