"""Pipeline for running wear-time detection algorithms on gait datasets."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
from tpcp import OptimizableParameter, OptimizablePipeline, make_action_safe, make_optimize_safe
from typing_extensions import Self

from mobgap.data.base import BaseGaitDataset
from mobgap.utils.conversions import to_body_frame
from mobgap.weartime.base import BaseWeartimeDetector, base_weartime_docfiller

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from mobgap.weartime.base import RecordingSampleCounts

_LOGGER = logging.getLogger(__name__)


def _conditionally_to_bf(data: pd.DataFrame, convert: bool) -> pd.DataFrame:
    if convert:
        return to_body_frame(data)
    return data


def _rss_mb() -> float | None:
    try:
        psutil = import_module("psutil")
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss / 1024**2)


class _TrainingDataFromDataset:
    def __init__(self, dataset: BaseGaitDataset, *, convert_to_body_frame: bool) -> None:
        self.dataset = dataset
        self.convert_to_body_frame = convert_to_body_frame

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        for datapoint_index, datapoint in enumerate(self.dataset):
            _LOGGER.debug(
                "Loading wear-time training datapoint %s: group=%s, rss_mb=%s",
                datapoint_index,
                datapoint.group_label,
                _rss_mb(),
            )
            data = _conditionally_to_bf(datapoint.data_ss, self.convert_to_body_frame)
            reference_weartime = datapoint.reference_weartime_
            _LOGGER.debug(
                "Loaded wear-time training datapoint %s: n_samples=%s, n_reference_intervals=%s, rss_mb=%s",
                datapoint_index,
                len(data),
                len(reference_weartime),
                _rss_mb(),
            )
            yield data, reference_weartime

    def load_recording(self, datapoint_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        datapoint = self.dataset[datapoint_index]
        _LOGGER.debug(
            "Loading wear-time training datapoint %s: group=%s, rss_mb=%s",
            datapoint_index,
            datapoint.group_label,
            _rss_mb(),
        )
        data = _conditionally_to_bf(datapoint.data_ss, self.convert_to_body_frame)
        reference_weartime = datapoint.reference_weartime_
        _LOGGER.debug(
            "Loaded wear-time training datapoint %s: n_samples=%s, n_reference_intervals=%s, rss_mb=%s",
            datapoint_index,
            len(data),
            len(reference_weartime),
            _rss_mb(),
        )
        return data, reference_weartime


def _single_sampling_rate_hz_and_sample_counts(dataset: BaseGaitDataset) -> tuple[float, RecordingSampleCounts]:
    sampling_rates = []
    sample_counts = []
    for datapoint in dataset:
        sampling_rates.append(float(datapoint.sampling_rate_hz))
        try:
            n_samples = int(datapoint.n_samples)
        except AttributeError as exc:
            raise AttributeError(
                "Wear-time model self-optimization requires datasets to expose `n_samples` for each datapoint."
            ) from exc
        if n_samples < 0:
            raise ValueError("Wear-time model self-optimization requires non-negative `n_samples` values.")
        sample_counts.append(n_samples)

    if not sampling_rates:
        raise ValueError("Cannot self-optimize a wear-time detector on an empty dataset.")

    sampling_rate_hz = sampling_rates[0]
    if not all(np.isclose(sampling_rate, sampling_rate_hz) for sampling_rate in sampling_rates):
        raise ValueError("Wear-time model self-optimization requires all datapoints to use the same sampling rate.")

    return sampling_rate_hz, tuple(sample_counts)


@base_weartime_docfiller
class WtdEmulationPipeline(OptimizablePipeline[BaseGaitDataset]):
    """Run a wear-time detection algorithm on a single dataset datapoint.

    This wraps any wear-time detector and allows it to be evaluated or optimized through tpcp's validation and
    optimization utilities.

    Parameters
    ----------
    algo
        The wear-time detector that should be run/evaluated.
    convert_to_body_frame
        If True, the data will be converted to the body frame before running the algorithm.
        This is the default, as the current wear-time detector expects body-frame columns.
        If your data is already body-frame aligned or your algorithm supports sensor-frame input, set this to False.

    Attributes
    ----------
    %(weartime_list_)s
    algo_
        The wear-time detector instance with all results after running the algorithm.
        This can be helpful for debugging or further analysis.

    Notes
    -----
    All emulation pipelines pass available metadata of the dataset to the algorithm.
    This includes the recording metadata (``recording_metadata``) and the participant metadata
    (``participant_metadata``), which are passed as keyword arguments to the ``detect`` method of the algorithm.
    In addition, we pass the group label of the datapoint as ``dp_group`` to the algorithm.
    This is usually not required by algorithms, but it can be helpful for dummy algorithms and cache keys.

    For the ``self_optimize`` method, the pipeline first reads the sampling rate and ``n_samples`` metadata for each
    datapoint. It then passes a lazy, re-iterable sequence of ``(data, reference_weartime)`` tuples to the algorithm.
    This keeps recording data loading at the dataset iterator boundary instead of collecting all recordings in memory
    first.
    """

    algo: OptimizableParameter[BaseWeartimeDetector]
    convert_to_body_frame: bool

    algo_: BaseWeartimeDetector

    def __init__(self, algo: BaseWeartimeDetector, *, convert_to_body_frame: bool = True) -> None:
        self.algo = algo
        self.convert_to_body_frame = convert_to_body_frame

    @property
    def weartime_list_(self) -> pd.DataFrame:  # noqa: D102
        return self.algo_.weartime_list_

    @property
    def total_weartime_samples_(self) -> int:  # noqa: D102
        return self.algo_.total_weartime_samples_

    @property
    def total_weartime_min_(self) -> float:  # noqa: D102
        return self.algo_.total_weartime_min_

    @property
    def total_weartime_during_waking_min_(self) -> float:  # noqa: D102
        return self.algo_.total_weartime_during_waking_min_

    @make_action_safe
    def run(self, datapoint: BaseGaitDataset) -> Self:
        """Run the detector on a single datapoint."""
        single_sensor_imu_data = _conditionally_to_bf(datapoint.data_ss, self.convert_to_body_frame)
        sampling_rate_hz = datapoint.sampling_rate_hz

        kwargs = {
            **datapoint.recording_metadata,
            **datapoint.participant_metadata,
            "dp_group": datapoint.group_label,
            "sampling_rate_hz": sampling_rate_hz,
        }

        self.algo_ = self.algo.clone().detect(single_sensor_imu_data, **kwargs)

        return self

    @make_optimize_safe
    def self_optimize(self, dataset: BaseGaitDataset) -> Self:
        """Run a detector's internal optimization routine, if implemented."""
        sampling_rate_hz, recording_sample_counts = _single_sampling_rate_hz_and_sample_counts(dataset)
        training_data = _TrainingDataFromDataset(dataset, convert_to_body_frame=self.convert_to_body_frame)
        self.algo = self.algo.self_optimize(
            training_data,
            sampling_rate_hz=sampling_rate_hz,
            recording_sample_counts=recording_sample_counts,
        )

        return self


__all__ = ["WtdEmulationPipeline"]
