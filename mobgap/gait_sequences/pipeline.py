"""Helpful Pipelines to wrap the GSD algorithms for optimization and evaluation."""

from typing import Any

import pandas as pd
from tpcp import OptimizableParameter, OptimizablePipeline
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import invert_list_of_dicts
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.gait_sequences.base import BaseGsDetector, base_gsd_docfiller
from mobgap.utils.conversions import to_body_frame


def _conditionally_to_bf(data: pd.DataFrame, convert: bool) -> pd.DataFrame:
    if convert:
        return to_body_frame(data)
    return data


@base_gsd_docfiller
class GsdEmulationPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """Run a GSD algorithm in isolation on a Gait Dataset.

    This wraps any GSD algorithm and allows to apply it to a single datapoint of a Gait Dataset or optimize it
    based on a whole dataset.

    This pipeline can be used in combination with the ``tpcp.validate`` and ``tpcp.optimize`` modules to evaluate or
    improve the performance of a GSD algorithm.

    Parameters
    ----------
    algo
        The GSD algorithm that should be run/evaluated.
    convert_to_body_frame
        If True, the data will be converted to the body frame before running the algorithm.
        This is the default, as most algorithm expect the data in the body frame.
        If your data is explictly not aligned and your algorithm supports sensor frame/unaligned input you might want
        to set this to False.

    Attributes
    ----------
    %(gs_list_)s
    algo_
        The GSD algo instance with all results after running the algorithm.
        This can be helpful for debugging or further analysis.


    Notes
    -----
    All emulation pipelines pass available metadata of the dataset to the algorithm.
    This includes the recording metadata (``recording_metadata``) and the participant metadata
    (``participant_metadata``), which are passed as keyword arguments to the ``detect`` method of the algorithm.
    In addition, we pass the group label of the datapoint as ``dp_group`` to the algorithm.
    This is usually not required by algorithms (because this would mean that the algorithm changes behaviour based on
    the exact recording provided).
    However, it can be helpful when working with "dummy" algorithms, that simply return some fixed pre-defined results
    or to be used as cache key, when the algorithm has internal caching mechanisms.

    For the `self_optimize` method, we pass the same metadata to the algorithm, but each value is actually a list of
    values, one for each datapoint in the dataset.
    """

    algo: OptimizableParameter[BaseGsDetector]
    convert_to_body_frame: bool

    algo_: BaseGsDetector

    def __init__(self, algo: BaseGsDetector, *, convert_to_body_frame: bool = True) -> None:
        self.algo = algo
        self.convert_to_body_frame = convert_to_body_frame

    @property
    def gs_list_(self) -> pd.DataFrame:  # noqa: D102
        return self.algo_.gs_list_

    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
        """Run the pipeline on a single data point.

        This extracts the imu_data (``data_ss``) and the sampling rate (``sampling_rate_hz``) from the datapoint and
        uses the ``detect`` method of the GSD algorithm to detect the gait sequences.

        Parameters
        ----------
        datapoint
            A single datapoint of a Gait Dataset with reference information.

        Returns
        -------
        self
            The pipeline instance with the detected gait sequences stored in the ``gs_list_`` attribute.

        """
        single_sensor_imu_data = _conditionally_to_bf(datapoint.data_ss, self.convert_to_body_frame)
        sampling_rate_hz = datapoint.sampling_rate_hz

        kwargs = {
            "sampling_rate_hz": sampling_rate_hz,
            **datapoint.recording_metadata,
            **datapoint.participant_metadata,
            "dp_group": datapoint.group_label,
        }

        self.algo_ = self.algo.clone().detect(single_sensor_imu_data, **kwargs)

        return self

    def self_optimize(self, dataset: BaseGaitDatasetWithReference, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Run a "self-optimization" of a GSD-algorithm (if it implements the respective method).

        This method extracts all data and reference gait sequences from the dataset and uses them to optimize the
        algorithm by calling the ``self_optimize`` method of the GSD algorithm (if it is implemented).

        Note, that this is only useful for algorithms with "internal" optimization logic (i.e. ML-based algorithms).
        If you want to optimize the hyperparameters of the algorithm, you should use the ``tpcp.optimize`` module.

        Parameters
        ----------
        dataset
            A Gait Dataset with reference information.
        kwargs
            Additional parameters required for the optimization process.
            This will be passed to the ``self_optimize`` method of the GSD algorithm.

        Returns
        -------
        self
            The pipeline instance with the optimized GSD algorithm.

        """
        # TODO: This method is not really tested yet, as we don't have any ML based GSD algorithms.
        all_data = (_conditionally_to_bf(d.data_ss, self.convert_to_body_frame) for d in dataset)
        dp_kwargs = invert_list_of_dicts(
            {**d.recording_metadata, **d.participant_metadata, "dp_group": d.group_label} for d in dataset
        )
        reference_wbs = (d.reference_parameters_.wb_list for d in dataset)
        sampling_rate_hz = (d.sampling_rate_hz for d in dataset)

        all_kwargs = {"sampling_rate_hz": sampling_rate_hz, **dp_kwargs, **kwargs}

        self.algo.self_optimize(all_data, reference_wbs, **all_kwargs)

        return self


__all__ = ["GsdEmulationPipeline"]
