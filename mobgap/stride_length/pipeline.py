import warnings
from typing import Self

import pandas as pd
from tpcp import OptimizableParameter, OptimizablePipeline

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.initial_contacts import refine_gs
from mobgap.pipeline import GsIterator
from mobgap.stride_length.base import BaseSlCalculator
from mobgap.utils.conversions import to_body_frame
from mobgap.utils.df_operations import create_multi_groupby
from mobgap.utils.interpolation import naive_sec_paras_to_regions


class SlEmulationPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """"""

    algo: OptimizableParameter[BaseSlCalculator]

    stride_length_per_sec_: pd.DataFrame
    stride_length_per_stride_unfiltered_: pd.DataFrame
    stride_length_per_stride_: pd.DataFrame
    per_wb_algo_: dict[str, BaseSlCalculator]

    def __init__(self, algo: BaseSlCalculator) -> None:
        self.algo = algo

    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
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
            warnings.warn(f"No walking bouts found in the reference data. {kwargs['dp_group']}", RuntimeWarning)
            self.per_wb_algo_ = {}
            self.stride_length_per_sec_ = pd.DataFrame(
                {"stride_length_m": [], "sec_center_samples": [], "wb_id": []}
            ).set_index(["wb_id", "sec_center_samples"])
            self.stride_length_per_stride_ = pd.DataFrame(
                {"stride_length_m": [], "start": [], "end": [], "lr_label": [], "wb_id": [], "s_id": []}
            ).set_index(["wb_id", "s_id"])
            return self

        result_algo_list = {}
        for (wb, data), r in wb_iterator.iterate(to_body_frame(datapoint.data_ss), ref_paras.wb_list):
            r.ic_list = ref_paras.ic_list.loc[wb.id]

            # The WBs from all reference systems are usually already defined so that they start and end with an
            # initial contact.
            # So refinement should not be required.
            # We have it here in the pipeline, in case we use other data input in the future.
            refined_wb_list, refined_ic_list = refine_gs(r.ic_list)

            with wb_iterator.subregion(refined_wb_list) as ((refined_wb, refined_gs_data), rr):
                algo = self.algo.clone().calculate(refined_gs_data, refined_ic_list, **kwargs, current_gs=refined_wb)
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
