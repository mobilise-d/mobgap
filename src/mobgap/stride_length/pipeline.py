"""Pipeline for running a stride length algorithm on a Gait Dataset."""

import warnings

import pandas as pd
from tpcp import OptimizableParameter, OptimizablePipeline
from typing_extensions import Self

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.initial_contacts import refine_gs
from mobgap.pipeline import GsIterator, Region
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.utils.conversions import to_body_frame
from mobgap.utils.df_operations import create_multi_groupby
from mobgap.utils.interpolation import naive_sec_paras_to_regions


class SlEmulationPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """Run a stride length algorithm in isolation per gait sequence/WB on a Gait Dataset.

    This wraps any stride length algorithm and allows running the algorithm on a single datapoint of a Gait Dataset.
    The pipeline uses the reference data for all required inputs to the algorithm.

    The algorithm will be executed once for each walking bout in the reference data.
    The algorithm will be provided with the initial contacts from the reference data as direct input.

    Parameters
    ----------
    algo
        The stride length algorithm that should be run/evaluated.

    Attributes
    ----------
    stride_length_per_sec_
        Dataframe containing the stride length for each second of each WB.
        This is the combined output of all algorithm results run on each WB.
    stride_length_per_stride_
        The interpolated stride length for each stride of each WB.
        The reference system is used to define the strides.
        Only strides that are considered valid by the reference system (stride length not NaN) are considered.
        This means that this output can be compared directly to the reference data on a stride level.
    per_wb_algo_
        A dictionary containing the algorithm instance for each WB.
        Each algorithm instance contains the results for the respective WB.
        This might be used for further analysis or debugging.

    Notes
    -----
    All emulation pipelines pass available metadata of the dataset to the algorithm.
    This includes the recording metadata (``recording_metadata``) and the participant metadata
    (``participant_metadata``), which are passed as keyword arguments to the ``detect`` method of the algorithm.
    In addition, we pass the group label of the datapoint as ``dp_group`` and the ``current_gs_absolute`` that
    provides the actual start and end of a GS relative to the start of the recording (instead of relative to surrounding
    region of interest).
    These two pieces of information are usually not required by algorithms (because this would mean that the algorithm
    changes behaviour based on the exact recording provided, or where in the data a WB occurred).
    However, it can be helpful when working with "dummy" algorithms, that simply return some fixed pre-defined results
    or to be used as cache key, when the algorithm has internal caching mechanisms.

    """

    algo: OptimizableParameter[BaseSlCalculator]

    stride_length_per_sec_: pd.DataFrame
    stride_length_per_stride_: pd.DataFrame
    per_wb_algo_: dict[str, BaseSlCalculator]

    def __init__(self, algo: BaseSlCalculator) -> None:
        self.algo = algo

    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
        """Run the stride length algorithm on a single datapoint.

        Parameters
        ----------
        datapoint
            A single datapoint of a Gait Dataset with reference information.

        Returns
        -------
        self
            The pipeline instance with all result attributes populated.

        """
        self.datapoint = datapoint
        sampling_rate_hz = datapoint.sampling_rate_hz

        kwargs = {
            "sampling_rate_hz": sampling_rate_hz,
            **datapoint.recording_metadata,
            **datapoint.participant_metadata,
            "dp_group": datapoint.group_label,
        }

        wb_iterator = GsIterator()
        ref_paras = datapoint.reference_parameters_relative_to_wb_

        if len(ref_paras.wb_list) == 0:
            warnings.warn(
                f"No walking bouts found in the reference data. {kwargs['dp_group']}", RuntimeWarning, stacklevel=1
            )
            self.per_wb_algo_ = {}
            self.stride_length_per_sec_ = (
                pd.DataFrame({"stride_length_m": [], "sec_center_samples": [], "wb_id": []})
                .astype({"stride_length_m": float, "sec_center_samples": int, "wb_id": int})
                .set_index(["wb_id", "sec_center_samples"])
            )
            self.stride_length_per_stride_ = (
                ref_paras.stride_parameters[["start", "end", "length_m"]]
                .rename(columns={"length_m": "stride_length_m"})
                .copy()
            )
            return self

        result_algo_list = {}
        for (wb, _), r in wb_iterator.iterate(to_body_frame(datapoint.data_ss), ref_paras.wb_list):
            r.ic_list = ref_paras.ic_list.loc[wb.id]

            # The WBs from all reference systems are usually already defined so that they start and end with an
            # initial contact.
            # So refinement should not be required.
            # We have it here in the pipeline, in case we use other data input in the future.
            refined_wb_list, refined_ic_list = refine_gs(r.ic_list)

            with wb_iterator.subregion(refined_wb_list) as ((refined_wb, refined_gs_data), rr):
                # Not quite happy, that we have to pass the current-gs offset here, but I don't see an easy way to
                # still make this pipeline universally usable and support our revalidation dummy algos.
                current_wb_absolute = Region(wb.id, wb.start + refined_wb.start, wb.end + refined_wb.start)
                algo = self.algo.clone().calculate(
                    refined_gs_data,
                    refined_ic_list,
                    **kwargs,
                    current_gs=refined_wb,
                    current_gs_absolute=current_wb_absolute,
                )
                result_algo_list[wb.id] = algo
                rr.stride_length_per_sec = algo.stride_length_per_sec_

        # The SL algorithms provide outputs per second of each WB.
        # We further provide interpolated outputs per stride.
        # Instead of constructing our stride list from the ICs, we use the stride list provided by the reference system.
        # This should be the fairest possible comparison.
        # We only consider reference strides that are considered valid by the system.
        stride_length_per_sec = wb_iterator.results_.stride_length_per_sec.reset_index("r_gs_id", drop=True)
        # We need the "non-relative" reference parameters to get the stride list.
        stride_list = datapoint.reference_parameters_.stride_parameters

        stride_list = stride_list[~stride_list["length_m"].isna()][["start", "end", "lr_label"]]
        stride_list_with_approx_paras = create_multi_groupby(
            stride_list,
            stride_length_per_sec,
            "wb_id",
            group_keys=False,
        ).apply(naive_sec_paras_to_regions, sampling_rate_hz=sampling_rate_hz)

        self.per_wb_algo_ = result_algo_list
        self.stride_length_per_sec_ = stride_length_per_sec
        self.stride_length_per_stride_ = stride_list_with_approx_paras

        return self
