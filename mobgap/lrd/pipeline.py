from collections.abc import Iterator
from typing import Any, Union, Unpack

import pandas as pd
from sklearn.metrics import accuracy_score
from tpcp import OptimizableParameter, OptimizablePipeline
from tpcp.validate import NoAgg
from typing_extensions import Self

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.lrd.base import BaseLRDetector
from mobgap.pipeline import GsIterator, iter_gs


def _extract_ref_data(dataset: BaseGaitDatasetWithReference) -> Iterator[tuple[str, pd.DataFrame, float]]:
    for datapoint in dataset:
        ref_paras = datapoint.reference_parameters_relative_to_wb_
        sampling_rate_hz = datapoint.sampling_rate_hz
        for _, per_wb_ic in ref_paras.ic_list.groupby("wb_id"):
            yield per_wb_ic, per_wb_ic.drop("lr_label", axis=1), sampling_rate_hz


def _extract_data(dataset: BaseGaitDatasetWithReference) -> Iterator[pd.DataFrame]:
    for datapoint in dataset:
        ref_params = datapoint.reference_parameters_relative_to_wb_
        for _, data in iter_gs(datapoint.data_ss, ref_params.wb_list):
            yield data


class LrdPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """
    This class represents a pipeline for LrdUllrich that can be optimized.
    """

    ic_lr_list_: pd.DataFrame

    algo: OptimizableParameter[BaseLRDetector]
    per_gs_algo_: list[BaseLRDetector]

    def __init__(self, algo: BaseLRDetector) -> None:
        self.algo = algo

    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
        """
        Runs the pipeline on a datapoint.

        Args:
            datapoint (mobgap.data._example_data.LabExampleDataset): The datapoint to run the pipeline on.

        Returns
        -------
            LrdUllrich: The algorithm with results.
        """
        sampling_rate_hz = datapoint.sampling_rate_hz

        ref_paras = datapoint.reference_parameters_relative_to_wb_

        gs_iterator = GsIterator()

        # TODO: maybe do it properly and create a custom iter type
        result_algo_list = []
        for (gs, data), r in gs_iterator.iterate(datapoint.data_ss, ref_paras.wb_list):
            ref_ic_list = ref_paras.ic_list.loc[gs.id]
            algo = self.algo.clone().detect(data, ref_ic_list, sampling_rate_hz=sampling_rate_hz)
            result_algo_list.append(algo)
            r.ic_list = algo.ic_lr_list_

        self.per_gs_algo_ = result_algo_list
        self.ic_lr_list_ = gs_iterator.results_.ic_list

        return self

    def self_optimize(self, dataset: BaseGaitDatasetWithReference, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Run a "self-optimization" of an LRD-algorithm (if it implements the respective method).

        This method extracts the data_list, ic_list, and left-right label from each wb in each datapoint in the dataset,
        and then calls the `self_optimize` method of the algorithm with these lists.

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
            The pipeline instance with the optimized LRD algorithm.

        """
        # Note: This will pull all values from the generator into memory
        # This is fine, as we assume that the ICs are small and should fit into memory easily
        # But by getting all values at once, we avoid loading (potentially uncached) the same reference parameters
        # multiple times.
        all_reference, all_ics, all_sampling_rate_hz = zip(*_extract_ref_data(dataset))
        # However, for the data, we want to keep the generator, as it might be large
        all_data = _extract_data(dataset)

        # Note: there is no cloning here -> we actually want to modify the object
        self.algo.self_optimize(all_data, all_ics, all_reference, sampling_rate_hz=all_sampling_rate_hz, **kwargs)

        return self

    def score(self, datapoint: BaseGaitDatasetWithReference) -> Union[float, dict[str, float]]:
        predicted_lr_labels = self.safe_run(datapoint).ic_lr_list_

        ref_labels = datapoint.reference_parameters_.ic_list["lr_label"]

        combined = predicted_lr_labels.assign(ref_lr_label=ref_labels)

        # TODO: Are there other useful metrics?
        return {"accuracy": accuracy_score(ref_labels, predicted_lr_labels["lr_label"]), "raw_results": NoAgg(combined)}
