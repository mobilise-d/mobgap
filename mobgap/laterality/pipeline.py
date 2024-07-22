"""Helpful Pipelines to wrap the LRC algorithms for optimization and evaluation."""

from collections.abc import Iterator
from typing import Any, TypedDict

import pandas as pd
from sklearn.metrics import accuracy_score
from tpcp import OptimizableParameter, OptimizablePipeline
from tpcp.validate import NoAgg
from typing_extensions import Self, Unpack

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.laterality.base import BaseLRClassifier
from mobgap.pipeline import GsIterator, iter_gs
from mobgap.utils.conversions import to_body_frame


def _extract_ref_data(dataset: BaseGaitDatasetWithReference) -> Iterator[tuple[str, pd.DataFrame, float]]:
    for datapoint in dataset:
        ref_paras = datapoint.reference_parameters_relative_to_wb_
        sampling_rate_hz = datapoint.sampling_rate_hz
        for _, per_wb_ic in ref_paras.ic_list.groupby("wb_id"):
            yield per_wb_ic, per_wb_ic.drop("lr_label", axis=1), sampling_rate_hz


def _extract_data(dataset: BaseGaitDatasetWithReference) -> Iterator[pd.DataFrame]:
    for datapoint in dataset:
        ref_params = datapoint.reference_parameters_relative_to_wb_
        for _, data in iter_gs(to_body_frame(datapoint.data_ss), ref_params.wb_list):
            yield data


class _LrcScores(TypedDict):
    accuracy: float
    raw_results: NoAgg


class LrcEmulationPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """Run a LRC algorithm in isolation, using reference ICs as input.

    This pipeline can wrap any LR-classifier and run it on a datapoint of any valid dataset or optimize it across a
    full dataset.
    The LRC is called once per WB in the datapoint and the reference initial contacts are used as the ``ic_list`` input
    for the algorithm.

    This pipeline should be used when performing a "block-wise" evaluation of an LRC algorithm or when optimizing an
    LRC either using external (``tpcp.optimize``) or internal (``self_optimize``) optimization.

    Parameters
    ----------
    algo
        The LRC algorithm to be run in the pipeline.

    Attributes
    ----------
    ic_lr_list_
        A dataframe containing all ICs across all WBs of a datapoint with an additional column ``lr_label`` specifying
        the detected left/right label.
    per_wb_algo_
        A dict of the LRC algorithm instances run on each WB of the datapoint.
        The key is the wb-id.
        Each instance contains the reference to the data it was called with, the classified labels and potential
        additional debug information provided by the individual algorithm.

    Other Parameters
    ----------------
    datapoint
        The datapoint that was passed to the run method.

    """

    algo: OptimizableParameter[BaseLRClassifier]

    ic_lr_list_: pd.DataFrame
    per_wb_algo_: dict[str, BaseLRClassifier]

    def __init__(self, algo: BaseLRClassifier) -> None:
        self.algo = algo

    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
        """Run the pipeline on a single datapoint.

        This extracts the imu data (``data_ss``) and the reference initial contact per reference WB within the
        datapoint and then calls the ``detect`` method of the algorithm once per WB.

        Parameters
        ----------
        datapoint
            A single datapoint of a Gait Dataset with reference information.

        Returns
        -------
        self
            The pipeline instance with the detected gait sequences stored in the ``gs_list_`` attribute.

        """
        self.datapoint = datapoint
        sampling_rate_hz = datapoint.sampling_rate_hz

        ref_paras = datapoint.reference_parameters_relative_to_wb_

        wb_iterator = GsIterator()

        # TODO: maybe do it properly and create a custom iter type
        result_algo_list = {}
        for (wb, data), r in wb_iterator.iterate(to_body_frame(datapoint.data_ss), ref_paras.wb_list):
            ref_ic_list = ref_paras.ic_list.loc[wb.id]
            algo = self.algo.clone().predict(
                data, ref_ic_list.drop("lr_label", axis=1, errors="ignore"), sampling_rate_hz=sampling_rate_hz
            )
            result_algo_list[wb.id] = algo
            r.ic_list = algo.ic_lr_list_

        self.per_wb_algo_ = result_algo_list
        self.ic_lr_list_ = wb_iterator.results_.ic_list

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

    def score(self, datapoint: BaseGaitDatasetWithReference) -> _LrcScores:
        """Score the pipeline on a single datapoint.

        This runs ``algo`` on the provided datapoint and returns the accuracy and the raw classified labels.

        This method should be used in combination with the scoring/validation methods available in ``tpcp.optimize``

        Parameters
        ----------
        datapoint
            A single datapoint of a Gait Dataset with reference information.

        Returns
        -------
        metrics
            A dictionary with relevant performance metrics

        """
        predicted_lr_labels = self.safe_run(datapoint).ic_lr_list_

        ref_labels = datapoint.reference_parameters_.ic_list["lr_label"]

        combined = predicted_lr_labels.assign(ref_lr_label=ref_labels)

        return {"accuracy": accuracy_score(ref_labels, predicted_lr_labels["lr_label"]), "raw_results": NoAgg(combined)}


__all__ = ["LrcEmulationPipeline"]
