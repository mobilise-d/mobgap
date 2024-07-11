from collections.abc import Iterator
from typing import Callable, Optional, Union

from sklearn.model_selection import BaseCrossValidator
from tpcp import Algorithm
from tpcp.optimize import BaseOptimize
from tpcp.validate import DatasetSplitter, NoAgg, cross_validate, validate
from typing_extensions import Self

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import measure_time, set_attrs_from_dict
from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.gait_sequences.pipeline import GsdEmulationPipeline


def gsd_evaluation_scorer(pipeline: GsdEmulationPipeline, datapoint: BaseGaitDatasetWithReference) -> dict:
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
    from mobgap.gait_sequences.evaluation import (
        calculate_matched_gsd_performance_metrics,
        calculate_unmatched_gsd_performance_metrics,
        categorize_intervals_per_sample,
    )

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


_gait_sequence_challenges_docfiller = make_filldoc(
    {
        "common_paras": """
    dataset
        A gait dataset with reference information.
        Evaluation is performed across all datapoints within the dataset.
    scoring
        A scoring function that evaluates the performance of the algorithm on a single datapoint.
        It should take a pipeline and a datapoint as input, run the pipeline on the datapoint and return a dictionary
        of performance metrics.
        These performance metrics are then aggregated across all datapoints.
    """,
        "timing_results": """
    start_datetime_utc_timestamp_
        The start time of the evaluation as UTC timestamp.
    start_datetime_
        The start time of the evaluation as human readable string.
    end_datetime_utc_timestamp_
        The end time of the evaluation as UTC timestamp.
    end_datetime_
        The end time of the evaluation as human readable string.
    runtime_s_
        The runtime of the evaluation in seconds.
        Note, that the runtime might not be exactly the difference between the start and the end time.
        The runtime is independently calculated using `time.perf_timer`.
    """,
    }
)


@_gait_sequence_challenges_docfiller
class GsdEvaluationCV(Algorithm):
    """Evaluation challenge for Gait Sequence Detection (GSD) algorithms using cross-validation.

    This class will use :func:`~tpcp.validate.cross_validate` to evaluate the performance of a GSD algorithm wrapped in
    a :class:`~mobgap.gait_sequences.pipeline.GsdEmulationPipeline` on a dataset with reference information.
    This is a suitable approach, when you want to evaluate and compare algorithms that are "trainable" in any way.
    This could be, because they are ML algorithms or because they have hyperparameters that can be optimized via
    Grid-Search.

    The cross validation parameters can be modified by the user to adapt them to a given dataset.

    Parameters
    ----------
    %(common_paras)s
    cv_iterator
        A valid cv_iterator.
        For complex CVs (e.g. stratified/grouped) this should be a :class:`~tpcp.validate.DatasetSplitter` instance.
        For more information see :func:`~tpcp.validate.cross_validate`.
    cv_params
        Dictionary with further parameters that are directly passed to :func:`~tpcp.validate.cross_validate`.
        This can overwrite all parameters except ``optimizable``, ``dataset``, ``scoring``, and ``cv``, which are
        directly set via the other parameters of this method.
        Typical usecase is to set ``n_jobs`` to activate multiprocessing.

    Other Parameters
    ----------------
    optimizer
        The tpcp optimizer passed to the ``run`` method.

    Attributes
    ----------
    cv_results_
        Dictionary with all results of the cross-validation.
        The results are returned by :func:`~tpcp.validate.cross_validate`.
        You can control what information is provided via ``cv_params``
    %(timing_results)s

    """

    _action_methods = "run"

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
    runtime_s_: float

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
        """Run the evaluation challenge.

        This will call the optimizer for each train set and evaluate the performance on each test set defined by the
        ``cv_iterator`` on the ``dataset``.

        Parameters
        ----------
        optimizer
            A valid tpcp optimizer that wraps a pipeline that is compatible with the provided dataset and scorer.
            Usually that should be an optimizer wrapping a
            :class:`~mobgap.gait_sequences.pipeline.GsdEmulationPipeline`.
            If you want to run without optimization, but still use the same test-folds, use
            :class:`~tpcp.optimize.DummyOptimize`:

            >>> from tpcp.optimize import DummyOptimize
            >>> from mobgap.gait_sequences import GsdIluz
            >>>
            >>> dummy_optimizer = DummyOptimize(
            ...     pipeline=GsdEmulationPipeline(GsdIluz()),
            ...     ignore_potential_user_error_warning=True,
            ... )
            >>> challenge = GsdEvaluationCV(dataset, cv_iterator, scoring=scoring)
            >>> challenge.run(dummy_optimizer)

        Returns
        -------
        self
            The instance of the class with the ``cv_results_`` attribute set to the results of the cross-validation.

        """
        self.optimizer = optimizer

        with measure_time() as timing_results:
            self.cv_results_ = cross_validate(
                optimizable=self.optimizer,
                dataset=self.dataset,
                scoring=self.scoring,
                cv=self.cv_iterator,
                **(self.cv_params or {}),
            )

        set_attrs_from_dict(self, timing_results, key_postfix="_")
        return self


@_gait_sequence_challenges_docfiller
class GsdEvaluation(Algorithm):
    """Evaluation challenge for Gait Sequence Detection (GSD) algorithms.

    This challenge applies the GSD algorithm wrapped in a :class:`~mobgap.gait_sequences.pipeline.GsdEmulationPipeline`
    to each datapoint in a dataset with reference information using :func:`~tpcp.validate.validate`.
    For each datapoint the provided scoring function is called and performance results are aggregated.

    This is a suitable approach, when you want to evaluate and compare algorithms that are not "trainable" in any way.
    For example, traditional algorithms or pre-trained models.
    Note, that if you are planning to compare algorithms that are trainable with non-trainable algorithms, you should
    use the :class:`~mobgap.gait_sequences.GsdEvaluationCV` for all of them.

    Parameters
    ----------
    %(common_paras)s
    validate_paras
        Dictionary with further parameters that are directly passed to :func:`~tpcp.validate.validate`.
        This can overwrite all parameters except ``pipeline``, ``dataset``, ``scoring``.
        Typical usecase is to set ``n_jobs`` to activate multiprocessing.

    Other Parameters
    ----------------
    pipeline
        The pipeline passed to the run method.

    Attributes
    ----------
    results_
        Dictionary with all results of the validation.
        The results are returned by :func:`~tpcp.validate.validate`.
        You can control what information is provided via ``validate_paras``
    %(timing_results)s

    """

    _action_methods = "run"

    dataset: BaseGaitDatasetWithReference
    scoring: Optional[Callable]

    pipeline: "GsdEmulationPipeline"

    results_: dict
    # timing results
    start_datetime_utc_timestamp_: float
    start_datetime_: str
    end_datetime_utc_timestamp_: float
    end_datetime_: str
    runtime_s_: float

    def __init__(
        self,
        dataset: BaseGaitDatasetWithReference,
        scoring: Optional[Callable] = gsd_evaluation_scorer,
        *,
        validate_paras: Optional[dict] = None,
    ) -> None:
        self.dataset = dataset
        self.scoring = scoring
        self.validate_paras = validate_paras

    def run(self, pipeline: "GsdEmulationPipeline") -> Self:
        """Run the evaluation challenge.

        This will call the pipeline for each datapoint in the dataset and evaluate the performance using the provided
        scoring function.

        Parameters
        ----------
        pipeline
            A valid pipeline that wraps a GSD algorithm that is compatible with the provided dataset and scorer.
            Usually that should be a :class:`~mobgap.gait_sequences.pipeline.GsdEmulationPipeline`.

        Returns
        -------
        self
            The instance of the class with the ``results_`` attribute set to the results of the validation.

        """
        self.pipeline = pipeline

        with measure_time() as timing_results:
            self.results_ = validate(
                pipeline=self.pipeline,
                dataset=self.dataset,
                scoring=self.scoring,
                **(self.validate_paras or {}),
            )

        set_attrs_from_dict(self, timing_results, key_postfix="_")
        return self
