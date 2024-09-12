from typing import Callable, Generic, Optional, Self, TypeVar

from tpcp import Algorithm, Pipeline
from tpcp.validate import validate, ScorerTypes

from mobgap._utils_internal.misc import measure_time, set_attrs_from_dict
from mobgap.data.base import BaseGaitDatasetWithReference

T = TypeVar("T", bound=Pipeline)


class Evaluation(Algorithm, Generic[T]):
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
    scoring: ScorerTypes

    pipeline: T

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
        scoring: ScorerTypes[T, BaseGaitDatasetWithReference],
        *,
        validate_paras: Optional[dict] = None,
    ) -> None:
        self.dataset = dataset
        self.scoring = scoring
        self.validate_paras = validate_paras

    def run(self, pipeline: T) -> Self:
        """Run the evaluation challenge.

        This will call the pipeline for each datapoint in the dataset and evaluate the performance using the provided
        scoring function.

        Parameters
        ----------
        pipeline
            A valid pipeline that is compatible with the provided dataset and scorer.

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
