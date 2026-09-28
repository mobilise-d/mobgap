"""Pipeline for running wear-time detection algorithms on gait datasets."""

from typing import Any

import pandas as pd
from tpcp import OptimizableParameter, OptimizablePipeline
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import invert_list_of_dicts
from mobgap.data.base import BaseGaitDataset
from mobgap.utils.conversions import to_body_frame
from mobgap.weartime.base import BaseWeartimeDetector, base_weartime_docfiller


def _conditionally_to_bf(data: pd.DataFrame, convert: bool) -> pd.DataFrame:
    if convert:
        return to_body_frame(data)
    return data


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

    For the `self_optimize` method, we pass the same metadata to the algorithm, but each value is a list of values,
    one for each datapoint in the dataset.
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

    def self_optimize(self, dataset: BaseGaitDataset, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Run a detector's internal optimization routine, if implemented."""
        all_data = (_conditionally_to_bf(d.data_ss, self.convert_to_body_frame) for d in dataset)
        dp_kwargs = invert_list_of_dicts(
            {**d.recording_metadata, **d.participant_metadata, "dp_group": d.group_label} for d in dataset
        )
        reference_weartime = (d.reference_weartime_ for d in dataset)
        sampling_rate_hz = (d.sampling_rate_hz for d in dataset)

        all_kwargs = {**dp_kwargs, **kwargs, "sampling_rate_hz": sampling_rate_hz}

        self.algo.self_optimize(all_data, reference_weartime, **all_kwargs)

        return self


__all__ = ["WtdEmulationPipeline"]
