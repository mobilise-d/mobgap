"""Helpful Pipelines to wrap the ICD algorithms for optimization and evaluation."""

import warnings

import pandas as pd
from tpcp import OptimizableParameter, OptimizablePipeline
from typing_extensions import Self

from mobgap._utils_internal.misc import MeasureTimeResults, timed_action_method
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.initial_contacts.base import BaseIcDetector, base_icd_docfiller
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame


def _conditionally_to_bf(data: pd.DataFrame, convert: bool) -> pd.DataFrame:
    if convert:
        return to_body_frame(data)
    return data


@base_icd_docfiller
class IcdEmulationPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """Run an ICD algorithm in isolation on a Gait Dataset.

    This wraps any ICD algorithm and allows to apply it to a single datapoint of a Gait Dataset or optimize it
    based on a whole dataset.

    This pipeline can be used in combination with the ``tpcp.validate`` and ``tpcp.optimize`` modules to evaluate or
    improve the performance of an ICD algorithm.

    Parameters
    ----------
    algo
        The ICD algorithm that should be run/evaluated.
    convert_to_body_frame
        If True, the data will be converted to the body frame before running the algorithm.
        This is the default, as most algorithm expect the data in the body frame.
        If your data is explicitly not aligned and your algorithm supports sensor frame/unaligned input you might want
        to set this to False.

    Attributes
    ----------
    %(ic_list_)s
    algo_
        The ICD algo instance with all results after running the algorithm.
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

    algo: OptimizableParameter[BaseIcDetector]
    convert_to_body_frame: bool

    per_wb_algo_: dict[str, BaseIcDetector]
    ic_list_: pd.DataFrame
    perf_: MeasureTimeResults

    def __init__(self, algo: BaseIcDetector, *, convert_to_body_frame: bool = True) -> None:
        self.algo = algo
        self.convert_to_body_frame = convert_to_body_frame

    @timed_action_method
    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
        """Run the pipeline on a single data point.

        This extracts the imu_data (``data_ss``) and the sampling rate (``sampling_rate_hz``) from the datapoint and
        uses the ``detect`` method of the ICD algorithm to detect the gait sequences.

        Parameters
        ----------
        datapoint
            A single datapoint of a Gait Dataset with reference information.

        Returns
        -------
        self
            The pipeline instance with the detected initial contacts stored in the ``ic_list_`` attribute.

        """
        imu_data = _conditionally_to_bf(datapoint.data_ss, self.convert_to_body_frame)
        sampling_rate_hz = datapoint.sampling_rate_hz

        kwargs = {
            "sampling_rate_hz": sampling_rate_hz,
            **datapoint.recording_metadata,
            **datapoint.participant_metadata,
            "dp_group": datapoint.group_label,
        }

        ref_paras = datapoint.reference_parameters_

        if len(ref_paras.wb_list) == 0:
            warnings.warn(
                f"No walking bouts found in the reference data. {kwargs['dp_group']}", RuntimeWarning, stacklevel=1
            )
            self.per_wb_algo_ = {}
            self.ic_list_ = pd.DataFrame({"wb_id": [], "step_id": [], "ic": []}).astype(
                {"wb_id": float, "step_id": int, "ic": int}
            )
            self.ic_list_ = self.ic_list_.set_index(["wb_id", "step_id"])
            return self
        wb_iterator = GsIterator()
        result_algo_list = {}
        for (wb, data), r in wb_iterator.iterate(imu_data, ref_paras.wb_list):
            algo = self.algo.clone().detect(data, **kwargs, current_gs=wb)
            result_algo_list[wb.id] = algo
            r.ic_list = algo.ic_list_
        self.per_wb_algo_ = result_algo_list
        self.ic_list_ = wb_iterator.results_.ic_list
        return self


__all__ = ["IcdEmulationPipeline"]
