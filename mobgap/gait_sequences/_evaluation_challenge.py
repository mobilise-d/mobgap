from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable, Optional, Union

from sklearn.model_selection import BaseCrossValidator
from tpcp import Algorithm
from tpcp.optimize import BaseOptimize
from tpcp.validate import DatasetSplitter, NoAgg, cross_validate, validate
from typing_extensions import Self

from mobgap._utils_internal.misc import measure_time, set_attrs_from_dict
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.gait_sequences.evaluation import (
    calculate_matched_gsd_performance_metrics,
    calculate_unmatched_gsd_performance_metrics,
    categorize_intervals_per_sample,
)

if TYPE_CHECKING:
    from mobgap.gait_sequences.pipeline import GsdEmulationPipeline


def gsd_evaluation_scorer(pipeline: "GsdEmulationPipeline", datapoint: BaseGaitDatasetWithReference) -> dict:
    """Evaluate the performance of a GSD algorithm on a single datapoint.

    This function is used to evaluate the performance of a GSD algorithm on a single datapoint.
    It calculates the performance metrics based on the detected gait sequences and the reference gait sequences.

    This is the default scoring functions for the GSD evaluation pipelines (``GsdEvaluation`` and ``GsdEvaluationCV``).

    Parameters
    ----------
    pipeline
        An instance of GSD emulation pipeline that wraps the algorithm that should be evaluated.
    datapoint
        The datapoint to be evaluated.

    Returns
    -------
    dict
        A dictionary containing the performance metrics.
        Note, that some results are wrapped in a ``NoAgg`` object or other aggregators.
        The results of this function are not expected to be parsed manually, but rather the function is expected to be
        used in the context of the :func:`~tpcp.validate.validate`/:func:`~tpcp.validate.cross_validate` functions or
        similar as scorer.
        This functions will aggregate the results and provide a summary of the performance metrics.

    """
    # Run the algorithm on the datapoint
    detected_gs_list = pipeline.safe_run(datapoint).gs_list_
    reference_gs_list = datapoint.reference_parameters_.wb_list[["start", "end"]]
    n_overall_samples = len(datapoint.data_ss)
    sampling_rate_hz = datapoint.sampling_rate_hz

    matches = categorize_intervals_per_sample(
        gsd_list_detected=detected_gs_list, gsd_list_reference=reference_gs_list, n_overall_samples=n_overall_samples
    )

    # Calculate the performance metrics
    performance_metrics = {
        **calculate_unmatched_gsd_performance_metrics(
            gsd_list_detected=detected_gs_list,
            gsd_list_reference=reference_gs_list,
            sampling_rate_hz=sampling_rate_hz,
        ),
        **calculate_matched_gsd_performance_metrics(matches),
        "detected": NoAgg(detected_gs_list),
        "reference": NoAgg(reference_gs_list),
    }

    return performance_metrics


class GsdEvaluationCV(Algorithm):
    dataset: BaseGaitDatasetWithReference
    cv_iterator: Optional[Union[DatasetSplitter, int, BaseCrossValidator, Iterator]]
    cv_params: Optional[dict]
    scoring: Optional[Callable]

    optimizer: BaseOptimize["GsdEmulationPipeline", BaseGaitDatasetWithReference]

    cv_results_: dict
    # timing results
    start_datetime_utc_timestamp_: float
    start_datetime_: str
    end_datetime_utc_timestamp_: float
    end_datetime_: str
    runtime_: float

    def __init__(
        self,
        dataset: BaseGaitDatasetWithReference,
        cv_iterator: Optional[Union[DatasetSplitter, int, BaseCrossValidator, Iterator]],
        scoring: Optional[Callable] = gsd_evaluation_scorer,
        *,
        cv_params: Optional[dict] = None,
    ) -> None:
        self.dataset = dataset
        self.cv_iterator = cv_iterator
        self.cv_params = cv_params
        self.scoring = scoring

    def run(self, optimizer: BaseOptimize["GsdEmulationPipeline", BaseGaitDatasetWithReference]) -> Self:
        self.optimizer = optimizer

        cv_params = {} if not self.cv_params else self.cv_params

        over_writable_cv_params = {
            "return_optimizer": True,
        }

        cv_params = {**over_writable_cv_params, **cv_params}

        with measure_time() as timing_results:
            self.cv_results_ = cross_validate(
                optimizable=self.optimizer,
                dataset=self.dataset,
                scoring=self.scoring,
                cv=self.cv_iterator,
                **cv_params,
            )

        set_attrs_from_dict(self, timing_results, key_postfix="_")
        return self


class GsdEvaluation(Algorithm):
    dataset: BaseGaitDatasetWithReference
    scoring: Optional[Callable]

    pipeline: "GsdEmulationPipeline"

    results_: dict
    # timing results
    start_datetime_utc_timestamp_: float
    start_datetime_: str
    end_datetime_utc_timestamp_: float
    end_datetime_: str
    runtime_: float

    def __init__(
        self, dataset: BaseGaitDatasetWithReference, scoring: Optional[Callable] = gsd_evaluation_scorer
    ) -> None:
        self.dataset = dataset
        self.scoring = scoring

    def run(self, pipeline: "GsdEmulationPipeline") -> Self:
        self.pipeline = pipeline

        with measure_time() as timing_results:
            self.results_ = validate(
                pipeline=self.pipeline,
                dataset=self.dataset,
                scoring=self.scoring,
            )

        set_attrs_from_dict(self, timing_results, key_postfix="_")
        return self
