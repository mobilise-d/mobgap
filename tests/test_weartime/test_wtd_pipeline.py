from collections.abc import Iterable
from datetime import time
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from typing_extensions import Self, Unpack

from mobgap.utils.conversions import to_body_frame
from mobgap.weartime.base import BaseWeartimeDetector, _unify_weartime_df
from mobgap.weartime.evaluation import wtd_final_agg, wtd_per_datapoint_score, wtd_score
from mobgap.weartime.pipeline import WtdEmulationPipeline


class GroupLabel(NamedTuple):
    participant_id: str
    recording_id: str


class DummyDatapoint:
    def __init__(
        self,
        *,
        data: pd.DataFrame,
        reference_weartime: pd.DataFrame,
        sampling_rate_hz: float,
        group_label: GroupLabel = GroupLabel("001", "rec_1"),
    ) -> None:
        self.data_ss = data
        self.reference_weartime_ = reference_weartime
        self.sampling_rate_hz = sampling_rate_hz
        self.group_label = group_label
        self.recording_metadata = {"measurement_condition": "laboratory", "recording_id": group_label.recording_id}
        self.participant_metadata = {"cohort": None, "height_m": None, "sensor_height_m": None}


class DummyWtd(BaseWeartimeDetector):
    def __init__(
        self,
        weartime_list: pd.DataFrame,
        *,
        waking_hours: tuple[time, time] = (time(0), time(0)),
        total_weartime_during_waking_min: Optional[float] = None,
    ) -> None:
        self.weartime_list = weartime_list
        self.waking_hours = waking_hours
        self.total_weartime_during_waking_min = total_weartime_during_waking_min

    def detect(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.detect_kwargs = kwargs
        self.weartime_list_ = _unify_weartime_df(self.weartime_list.copy())
        self.perf_ = {"runtime_s": 1.25}
        return self

    @property
    def total_weartime_during_waking_min_(self) -> float:
        if self.total_weartime_during_waking_min is not None:
            return self.total_weartime_during_waking_min
        return super().total_weartime_during_waking_min_


class MinimalWtd(BaseWeartimeDetector):
    def detect(self, data: pd.DataFrame, *, sampling_rate_hz: float, **kwargs: Unpack[dict[str, Any]]) -> Self:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.weartime_list_ = _intervals([(0, len(data))])
        return self


class DummyOptimizableWtd(DummyWtd):
    def self_optimize(
        self,
        training_data: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
        *,
        sampling_rate_hz: float,
    ) -> BaseWeartimeDetector:
        records = list(training_data)
        self.weartime_list = records[0][1]
        self.total_weartime_during_waking_min = sampling_rate_hz + len(records[0][0])
        return self


class FailingOptimizableWtd(DummyOptimizableWtd):
    def self_optimize(
        self,
        training_data: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
        *,
        sampling_rate_hz: float,
    ) -> BaseWeartimeDetector:
        super().self_optimize(training_data, sampling_rate_hz=sampling_rate_hz)
        raise RuntimeError("training failed")


def _intervals(intervals: list[tuple[int, int]], index_name: str = "wt_id") -> pd.DataFrame:
    return pd.DataFrame(intervals, columns=["start", "end"]).rename_axis(index_name)


def _sensor_frame_data(n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "acc_x": [0.0] * n_samples,
            "acc_y": [0.0] * n_samples,
            "acc_z": [0.0] * n_samples,
            "gyr_x": [0.0] * n_samples,
            "gyr_y": [0.0] * n_samples,
            "gyr_z": [0.0] * n_samples,
        }
    )


def test_wtd_emulation_pipeline_converts_to_body_frame_by_default():
    data = _sensor_frame_data(3)
    datapoint = DummyDatapoint(data=data, reference_weartime=_intervals([(0, 3)]), sampling_rate_hz=10.0)
    datapoint.recording_metadata["sampling_rate_hz"] = 999.0

    pipeline = WtdEmulationPipeline(DummyWtd(_intervals([(1, 2)]))).run(datapoint)

    assert_frame_equal(pipeline.algo_.data, to_body_frame(data))
    assert pipeline.algo_.sampling_rate_hz == 10.0
    assert pipeline.algo_.detect_kwargs["recording_id"] == "rec_1"
    assert pipeline.algo_.detect_kwargs["dp_group"] == GroupLabel("001", "rec_1")
    assert pipeline.total_weartime_samples_ == 1
    assert pipeline.total_weartime_min_ == pytest.approx(1 / 600)
    assert pipeline.total_weartime_during_waking_min_ == pytest.approx(1 / 600)
    assert not hasattr(pipeline, "total_weartime_minutes_")
    assert not hasattr(pipeline, "total_weartime_hours_")
    assert not hasattr(pipeline, "total_weartime_hours_during_waking_")


def test_pipeline_accepts_minimal_detector_with_default_waking_hours():
    datapoint = DummyDatapoint(
        data=_sensor_frame_data(60), reference_weartime=_intervals([(0, 60)]), sampling_rate_hz=1.0
    )

    pipeline = WtdEmulationPipeline(MinimalWtd()).safe_run(datapoint)

    assert pipeline.total_weartime_min_ == 1.0
    assert pipeline.total_weartime_during_waking_min_ == 0.0


def test_wtd_emulation_pipeline_optimizes_with_paired_training_records():
    datapoints = [
        DummyDatapoint(
            data=_sensor_frame_data(3),
            reference_weartime=_intervals([(0, 3)]),
            sampling_rate_hz=10.0,
        ),
        DummyDatapoint(
            data=_sensor_frame_data(4),
            reference_weartime=_intervals([(1, 4)]),
            sampling_rate_hz=10.0,
            group_label=GroupLabel("002", "rec_2"),
        ),
    ]

    algo = DummyOptimizableWtd(_intervals([]))
    other_pipeline = WtdEmulationPipeline(algo)
    pipeline = WtdEmulationPipeline(algo).self_optimize(datapoints)

    assert pipeline.algo is not algo
    assert other_pipeline.algo is algo
    assert algo.weartime_list.empty
    assert algo.total_weartime_during_waking_min is None
    assert pipeline.algo.total_weartime_during_waking_min == 13.0
    assert_frame_equal(pipeline.algo.weartime_list, datapoints[0].reference_weartime_)


def test_failed_optimization_preserves_supplied_detector():
    algo = FailingOptimizableWtd(_intervals([]))
    pipeline = WtdEmulationPipeline(algo)
    datapoints = [
        DummyDatapoint(data=_sensor_frame_data(3), reference_weartime=_intervals([(0, 3)]), sampling_rate_hz=10.0)
    ]

    with pytest.raises(RuntimeError, match="training failed"):
        pipeline.self_optimize(datapoints)

    assert pipeline.algo is algo
    assert algo.weartime_list.empty
    assert algo.total_weartime_during_waking_min is None


def test_optimization_rejects_uncertain_ground_truth():
    datapoint = DummyDatapoint(
        data=_sensor_frame_data(4), reference_weartime=_intervals([(0, 2)]), sampling_rate_hz=1.0
    )
    datapoint.reference_uncertain_ = _intervals([(2, 4)])

    with pytest.raises(ValueError, match="uncertain ground truth"):
        WtdEmulationPipeline(DummyOptimizableWtd(_intervals([]))).self_optimize([datapoint])


def test_wtd_score_counts_half_open_samples_and_minute_durations():
    data = _sensor_frame_data(120)
    datapoint = DummyDatapoint(
        data=data,
        reference_weartime=_intervals([(0, 119)], index_name="weartime_id"),
        sampling_rate_hz=1.0,
    )
    pipeline = WtdEmulationPipeline(
        DummyWtd(_intervals([(60, 119)]), waking_hours=(time(0, 1), time(0, 2)), total_weartime_during_waking_min=0.5)
    )

    scores = wtd_per_datapoint_score(pipeline, datapoint, zero_division=0)

    assert scores["tp_samples"] == 59
    assert scores["fn_samples"] == 60
    assert scores["fp_samples"] == 0
    assert scores["tn_samples"] == 1
    assert sum(scores[f"{kind}_samples"] for kind in ("tp", "fp", "fn", "tn")) == len(data)
    assert scores["reference_weartime_min"] == pytest.approx(119 / 60)
    assert scores["detected_weartime_min"] == pytest.approx(59 / 60)
    assert scores["weartime_error_min"] == pytest.approx(-1.0)
    assert scores["waking_reference_weartime_min"] == pytest.approx(59 / 60)
    assert scores["waking_detected_weartime_min"] == pytest.approx(0.5)
    assert scores["waking_weartime_error_min"] == pytest.approx(0.5 - 59 / 60)
    assert scores["runtime_s"] == 1.25
    assert_frame_equal(scores["detected"].get_value(), _intervals([(60, 119)]))
    assert_frame_equal(scores["reference"].get_value(), _intervals([(0, 119)], index_name="weartime_id"))


@pytest.mark.parametrize("day", ["2026-03-29", "2026-10-25"])
def test_wtd_score_accepts_uk_local_days_across_dst(day):
    start = pd.Timestamp(day, tz="Europe/London")
    end = start + pd.DateOffset(days=1)
    index = pd.date_range(start, end, freq="s", inclusive="left")
    data = _sensor_frame_data(len(index))
    data.index = index
    datapoint = DummyDatapoint(
        data=data,
        reference_weartime=_intervals([(0, len(data))], index_name="weartime_id"),
        sampling_rate_hz=1.0,
    )
    pipeline = WtdEmulationPipeline(DummyWtd(_intervals([(0, len(data))]), waking_hours=(time(7), time(22))))

    scores = wtd_per_datapoint_score(pipeline, datapoint, zero_division=0)

    assert scores["waking_reference_weartime_min"] == 900
    assert scores["waking_detected_weartime_min"] == 900


def test_wtd_score_counts_all_nonwear_samples():
    data = _sensor_frame_data(3)
    datapoint = DummyDatapoint(data=data, reference_weartime=_intervals([]), sampling_rate_hz=1.0)
    pipeline = WtdEmulationPipeline(DummyWtd(_intervals([]), waking_hours=(time(0), time(0, 1))))

    scores = wtd_per_datapoint_score(pipeline, datapoint, zero_division=0)

    assert scores["tn_samples"] == 3
    assert scores["tp_samples"] == scores["fp_samples"] == scores["fn_samples"] == 0
    assert scores["reference_weartime_min"] == 0


def test_wtd_score_excludes_uncertain_samples_from_all_metrics():
    data = _sensor_frame_data(120)
    datapoint = DummyDatapoint(data=data, reference_weartime=_intervals([(0, 60)]), sampling_rate_hz=1.0)
    datapoint.reference_uncertain_ = _intervals([(60, 120)])
    pipeline = WtdEmulationPipeline(DummyWtd(_intervals([(0, 120)]), waking_hours=(time(0), time(0, 2))))

    scores = wtd_per_datapoint_score(pipeline, datapoint, zero_division=0)

    assert scores["tp_samples"] == 60
    assert scores["fp_samples"] == 0
    assert scores["detected_weartime_min"] == 1
    assert scores["waking_detected_weartime_min"] == 1
    assert scores["weartime_error_min"] == 0


def test_scored_interval_fragments_keep_original_detector_ids():
    datapoint = DummyDatapoint(
        data=_sensor_frame_data(30), reference_weartime=_intervals([(0, 30)]), sampling_rate_hz=1.0
    )
    datapoint.reference_uncertain_ = _intervals([(10, 20)])
    detected = _intervals([(0, 30)])
    detected.index = pd.Index([42], name="wt_id")

    scores = wtd_per_datapoint_score(
        WtdEmulationPipeline(DummyWtd(detected, waking_hours=(time(0), time(0, 1)))), datapoint, zero_division=0
    )

    assert_frame_equal(scores["detected"].get_value(), detected)
    assert scores["detected_scored"].get_value().index.tolist() == [42, 42]
    assert scores["detected_scored"].get_value()[["start", "end"]].to_numpy().tolist() == [[0, 10], [20, 30]]


def test_uncertainty_outside_intervals_keeps_nonconsecutive_source_ids():
    detected = _intervals([(0, 10), (20, 30)])
    detected.index = pd.Index([7, 42], name="wt_id")
    datapoint = DummyDatapoint(data=_sensor_frame_data(30), reference_weartime=detected, sampling_rate_hz=1.0)
    datapoint.reference_uncertain_ = _intervals([(12, 18)])

    scores = wtd_per_datapoint_score(
        WtdEmulationPipeline(DummyWtd(detected, waking_hours=(time(0), time(0, 1)))), datapoint, zero_division=0
    )

    assert_frame_equal(scores["detected_scored"].get_value(), detected)
    assert scores["reference_scored"].get_value().index.tolist() == [7, 42]


def test_fully_uncertain_waking_window_does_not_bias_waking_average():
    uncertain_day = DummyDatapoint(
        data=_sensor_frame_data(7200),
        reference_weartime=_intervals([(0, 3600)]),
        sampling_rate_hz=1.0,
        group_label=GroupLabel("010", "uncertain_waking"),
    )
    uncertain_day.reference_uncertain_ = _intervals([(3600, 7200)])
    labeled_day = DummyDatapoint(
        data=_sensor_frame_data(7200),
        reference_weartime=_intervals([(0, 7200)]),
        sampling_rate_hz=1.0,
    )
    pipeline = WtdEmulationPipeline(DummyWtd(_intervals([(0, 7200)]), waking_hours=(time(1), time(2))))

    scores = wtd_per_datapoint_score(pipeline, uncertain_day, zero_division=0)
    aggregate, single = wtd_score(pipeline, [uncertain_day, labeled_day])

    assert np.isnan(scores["waking_weartime_error_min"])
    assert np.isnan(single["waking_detected_weartime_min"][0])
    assert aggregate["waking_reference_weartime_min"] == 60
    assert aggregate["combined__waking_reference_weartime_min"] == 60


def test_all_uncertain_dataset_has_no_combined_performance_rates():
    datapoint = DummyDatapoint(data=_sensor_frame_data(60), reference_weartime=_intervals([]), sampling_rate_hz=1.0)
    datapoint.reference_uncertain_ = _intervals([(0, 60)])

    aggregate, _ = wtd_score(
        WtdEmulationPipeline(DummyWtd(_intervals([(0, 60)]), waking_hours=(time(0), time(0, 1)))),
        [datapoint],
    )

    assert aggregate["combined__tp_samples"] == 0
    assert np.isnan(aggregate["combined__accuracy"])
    assert np.isnan(aggregate["combined__weartime_error_min"])
    assert np.isnan(aggregate["combined__waking_weartime_error_min"])


@pytest.mark.parametrize("uncertain_first", [False, True])
def test_fully_uncertain_day_does_not_bias_mean_or_combined_scores(uncertain_first):
    certain = DummyDatapoint(
        data=_sensor_frame_data(60), reference_weartime=_intervals([(0, 60)]), sampling_rate_hz=1.0
    )
    uncertain = DummyDatapoint(
        data=_sensor_frame_data(60),
        reference_weartime=_intervals([]),
        sampling_rate_hz=1.0,
        group_label=GroupLabel("010", "uncertain"),
    )
    uncertain.reference_uncertain_ = _intervals([(0, 60)])
    pairs = [(certain, WtdEmulationPipeline(DummyWtd(_intervals([(0, 60)]))))]
    uncertain_pair = (uncertain, WtdEmulationPipeline(DummyWtd(_intervals([(0, 60)]))))
    pairs.insert(0 if uncertain_first else 1, uncertain_pair)
    scores = [wtd_per_datapoint_score(pipeline, datapoint, zero_division=0) for datapoint, pipeline in pairs]
    single_results = {
        key: [score[key].get_value() if hasattr(score[key], "get_value") else score[key] for score in scores]
        for key in scores[0]
    }

    assert np.nanmean(single_results["accuracy"]) == 1
    assert np.nanmean(single_results["detected_weartime_min"]) == 1
    combined, raw = wtd_final_agg({}, single_results, pairs[0][1], [datapoint for datapoint, _ in pairs])
    assert combined["combined__accuracy"] == 1
    assert combined["combined__detected_weartime_min"] == 1
    assert raw["raw__detected"].index.names == ["participant_id", "recording_id", "wt_id"]
    aggregate, _ = wtd_score(pairs[0][1], [datapoint for datapoint, _ in pairs])
    assert aggregate["accuracy"] == 1
    assert aggregate["detected_weartime_min"] == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_wtd_score_combines_half_open_matches_across_datapoints(reverse):
    first = DummyDatapoint(
        data=_sensor_frame_data(3),
        reference_weartime=_intervals([(0, 3)]),
        sampling_rate_hz=1.0,
    )
    second = DummyDatapoint(
        data=_sensor_frame_data(2),
        reference_weartime=_intervals([]),
        sampling_rate_hz=1.0,
        group_label=GroupLabel("002", "rec_2"),
    )
    first_pipeline = WtdEmulationPipeline(DummyWtd(_intervals([(1, 2)]), waking_hours=(time(0), time(0, 1))))
    second_pipeline = WtdEmulationPipeline(DummyWtd(_intervals([(0, 1)]), waking_hours=(time(0), time(0, 1))))
    pairs = [(first, first_pipeline), (second, second_pipeline)]
    if reverse:
        pairs.reverse()
    scores = [wtd_per_datapoint_score(pipeline, datapoint, zero_division=0) for datapoint, pipeline in pairs]
    single_results = {
        key: [score[key].get_value() if hasattr(score[key], "get_value") else score[key] for score in scores]
        for key in scores[0]
    }

    combined, raw = wtd_final_agg({}, single_results, first_pipeline, [datapoint for datapoint, _ in pairs])

    assert [combined[f"combined__{kind}_samples"] for kind in ("tp", "fp", "fn", "tn")] == [1, 1, 2, 1]
    assert combined["combined__reference_weartime_min"] == pytest.approx(3 / 60)
    assert raw["raw__reference_waking"].index.names == ["participant_id", "recording_id", "weartime_id"]


@pytest.mark.parametrize("waking_hours, expected_minutes", [((time(7), time(22)), 0), ((time(0, 5), time(0, 20)), 5)])
def test_short_recording_clips_detected_and_reference_waking_time(waking_hours, expected_minutes):
    intervals = _intervals([(0, 600)])
    datapoint = DummyDatapoint(data=_sensor_frame_data(600), reference_weartime=intervals, sampling_rate_hz=1.0)
    pipeline = WtdEmulationPipeline(DummyWtd(intervals, waking_hours=waking_hours))

    scores = wtd_per_datapoint_score(pipeline, datapoint, zero_division=0)

    assert scores["waking_detected_weartime_min"] == expected_minutes
    assert scores["waking_reference_weartime_min"] == expected_minutes
    assert scores["waking_weartime_error_min"] == 0
