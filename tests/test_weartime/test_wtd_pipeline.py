from typing import Any, NamedTuple, Optional

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from typing_extensions import Self, Unpack

from mobgap.utils.conversions import to_body_frame
from mobgap.weartime.base import BaseWeartimeDetector, _unify_weartime_df
from mobgap.weartime.evaluation import wtd_per_datapoint_score
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
        waking_hours_min: tuple[int, int] = (0, 24 * 60),
        total_weartime_during_waking_min: Optional[float] = None,
    ) -> None:
        self.weartime_list = weartime_list
        self.waking_hours_min = waking_hours_min
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


def test_wtd_score_uses_gsd_sample_counts_and_minute_durations():
    data = _sensor_frame_data(120)
    datapoint = DummyDatapoint(
        data=data,
        reference_weartime=_intervals([(0, 119)], index_name="weartime_id"),
        sampling_rate_hz=1.0,
    )
    pipeline = WtdEmulationPipeline(
        DummyWtd(_intervals([(60, 119)]), waking_hours_min=(1, 2), total_weartime_during_waking_min=0.5)
    )

    scores = wtd_per_datapoint_score(pipeline, datapoint, zero_division=0)

    assert scores["tp_samples"] == 60
    assert scores["fn_samples"] == 61
    assert scores["fp_samples"] == 0
    assert scores["reference_weartime_min"] == pytest.approx(2.0)
    assert scores["detected_weartime_min"] == pytest.approx(59 / 60)
    assert scores["weartime_error_min"] == pytest.approx(59 / 60 - 2.0)
    assert scores["waking_reference_weartime_min"] == pytest.approx(1.0)
    assert scores["waking_detected_weartime_min"] == pytest.approx(0.5)
    assert scores["waking_weartime_error_min"] == pytest.approx(-0.5)
    assert scores["runtime_s"] == 1.25
    assert_frame_equal(scores["detected"].get_value(), _intervals([(60, 119)]))
    assert_frame_equal(scores["reference"].get_value(), _intervals([(0, 119)], index_name="weartime_id"))
