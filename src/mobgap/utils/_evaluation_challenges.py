import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, Literal, Optional, TypeVar, Union

import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from tpcp import Algorithm, Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import DatasetSplitter, ScorerTypes, cross_validate, validate
from typing_extensions import Self

from mobgap._docutils import make_filldoc
from mobgap._utils_internal.misc import MeasureTimeResults, measure_time, timer_doc_filler
from mobgap.data.base import BaseGaitDatasetWithReference

T = TypeVar("T", bound=Pipeline)

evaluation_challenges_docfiller = make_filldoc(
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
    """
    }
    | timer_doc_filler._dict,
)


@evaluation_challenges_docfiller
class Evaluation(Algorithm, Generic[T]):
    """Gneric Evaluation challenge for all algorithms.

    This challenge wraps any valid gait pipeline together with a scoring function and runs and scores it on a dataset.

    This is a suitable approach, when you want to evaluate and compare algorithms that are not "trainable" in any way.
    For example, traditional algorithms or pre-trained models.
    Note, that if you are planning to compare algorithms that are trainable with non-trainable algorithms, you should
    use the :class:`~mobgap.utils.evaluation.EvaluationCV` for all of them.

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
    %(perf_)s

    """

    _action_methods = "run"

    dataset: BaseGaitDatasetWithReference
    scoring: ScorerTypes[T, BaseGaitDatasetWithReference]

    pipeline: T

    results_: dict
    perf_: MeasureTimeResults

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

        self.perf_ = timing_results
        return self

    def get_single_results_as_df(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Get the results as a pandas DataFrame.

        This will return the results as a pandas DataFrame with the columns specified in the ``columns`` parameter.
        If no columns are specified, all columns are returned.
        We exclude `single__raw__` columns, as they are by convention reserved for the direct output of the pipeline and
        usually don't make sense to view together with the single results.

        This will provide as one row per data label.

        Parameters
        ----------
        columns
            List of columns that should be included in the DataFrame.
            These need to be specified WITHOUT the "single__" prefix.
            (e.g. `f1_score` instead of `single__f1_score`).
            If not specified, all columns are included.

        Returns
        -------
        pd.DataFrame
            The results as a pandas DataFrame.

        """
        result = pd.DataFrame(self.results_)
        if columns is None:
            columns = [c for c in result.columns if c.startswith("single__") and not c.startswith("single__raw__")]
        else:
            columns = [f"single__{c}" for c in columns]
        relevant_cols = ["data_labels", *columns]
        result = result[relevant_cols]
        result = result.explode(relevant_cols).set_index("data_labels", append=True)
        result.columns = [c.split("__", 1)[-1] for c in result.columns]
        result.index = pd.MultiIndex.from_tuples(
            [(fold, *data_label) for fold, data_label in result.index], names=("fold", *result.index[0][1]._fields)
        )
        return result

    def get_aggregated_results_as_df(self) -> pd.DataFrame:
        """Get the aggregated results as a pandas DataFrame.

        This will return all `agg__` columns that the scorer returned (see `results_` attribute) as a pandas dataframe.

        The returned Df just has a single row with the index `0` and each column represents one aggregated values.
        This shape is used, to provide equivalent output to the results of the cross-validation.

        Returns
        -------
        pd.DataFrame
            The results as a pandas DataFrame.

        """
        result = pd.DataFrame(self.results_)
        relevant_cols = [c for c in result.columns if c.startswith("agg__")]
        result = result[relevant_cols]
        result.columns = [c.split("__", 1)[-1] for c in result.columns]
        result = result.rename_axis(index="fold")
        return result

    def get_raw_results(self) -> dict:
        """Get the raw results of the cross-validation.

        Get the direct output of the algorithms.
        These are usually handed down through the `single__raw__` parameters of the scoring output.

        The exact structure of the results depends on the scorer and the optimizer used.
        Usually, outputs are provided as pandas dataframes.

        If the individual outputs are dataframes, they are concatenated along the `cv_fold` axis.
        Otherwise, they are simply returned as a list, where each element represents the output of one cv-fold.

        Returns
        -------
        dict
            Raw algorithm results from teh cross-validation.

        """
        out = {}
        for k, v in self.results_.items():
            if not k.startswith("single__raw__"):
                continue
            key = k.split("__", 2)[-1]
            if isinstance(v[0], pd.DataFrame):
                out[key] = pd.concat(v, keys=range(len(v)), names=["fold", *v[0].index.names])
            else:
                out[key] = v

        return out


@evaluation_challenges_docfiller
class EvaluationCV(Algorithm, Generic[T]):
    """Generic Evaluation challenge for all algorithms using a cross-validation for scoring.

    This class will use :func:`~tpcp.validate.cross_validate` to evaluate the performance of a pipeline on a dataset
    with reference information.
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
    results_
        Dictionary with all results of the cross-validation.
        The results are returned by :func:`~tpcp.validate.cross_validate`.
        You can control what information is provided via ``cv_params``
    %(perf_)s

    """

    _action_methods = "run"

    dataset: BaseGaitDatasetWithReference
    cv_iterator: Optional[Union[DatasetSplitter, int, BaseCrossValidator, Iterator]]
    cv_params: Optional[dict]
    scoring: ScorerTypes[T, BaseGaitDatasetWithReference]

    optimizer: BaseOptimize[T, BaseGaitDatasetWithReference]

    results_: dict
    perf_: MeasureTimeResults

    def __init__(
        self,
        dataset: BaseGaitDatasetWithReference,
        scoring: ScorerTypes[T, BaseGaitDatasetWithReference],
        cv_iterator: Optional[Union[DatasetSplitter, int, BaseCrossValidator, Iterator]],
        *,
        cv_params: Optional[dict] = None,
    ) -> None:
        self.dataset = dataset
        self.cv_iterator = cv_iterator
        self.cv_params = cv_params
        self.scoring = scoring

    def run(self, optimizer: BaseOptimize[T, BaseGaitDatasetWithReference]) -> Self:
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
            >>> challenge = EvaluationCV(
            ...     dataset,
            ...     scoring,
            ...     cv_iterator,
            ... )
            >>> challenge.run(dummy_optimizer)

        Returns
        -------
        self
            The instance of the class with the ``results_`` attribute set to the results of the cross-validation.

        """
        self.optimizer = optimizer

        with measure_time() as timing_results:
            self.results_ = cross_validate(
                optimizable=self.optimizer,
                dataset=self.dataset,
                scoring=self.scoring,
                cv=self.cv_iterator,
                **(self.cv_params or {}),
            )

        self.perf_ = timing_results
        return self

    def get_single_results_as_df(
        self, columns: Optional[list[str]] = None, *, group: Literal["train", "test"] = "test"
    ) -> pd.DataFrame:
        """Get the results as a pandas DataFrame.

        This will return the results as a pandas DataFrame with the columns specified in the ``columns`` parameter.
        If no columns are specified, all columns are returned.

        We exclude `single__raw__` columns, as they are by convention reserved for the direct output of the pipeline
        and usually don't make sense to view together with the other single results.

        This will provide as one row per data label of all datapoints across all cv-folds.
        Be aware, that this means, that these results were potentially generated with different models or
        hyperparameters (depending on what you are optimizing).

        .. warning:: When using `group="train"`, you will likely get duplicated rows in the results, as the same
            datapoints were used in multiple cv-folds as training data.
            You should remove these duplicates depending on your application.

        Parameters
        ----------
        columns
            List of columns that should be included in the DataFrame.
            These need to be specified WITHOUT the "single__" and the `test/train__` prefix.
            (e.g. `f1_score` instead of `test__single__f1_score`).
            If not specified, all columns are included.
        group
            Whether to return the results for the test or the train set.
            Note, that the train results might only be available, if you passed `return_train_scores=True` to the
            `cv_params` of the `EvaluationCv` instance.

        Returns
        -------
        pd.DataFrame
            The results as a pandas DataFrame.

        """
        result = pd.DataFrame(self.results_)
        if columns is None:
            columns = [
                c
                for c in result.columns
                if c.startswith(f"{group}__single__") and not c.startswith(f"{group}__single__raw__")
            ]
        else:
            columns = [f"{group}__single__{c}" for c in columns]
        relevant_cols = [f"{group}__data_labels", *columns]
        result = result[relevant_cols]
        result = result.explode(relevant_cols)
        result.columns = [c.split("__", 2)[-1] for c in result.columns]
        result = result.set_index("data_labels", append=True)
        result.index = pd.MultiIndex.from_tuples(
            [(fold, *data_label) for fold, data_label in result.index], names=("fold", *result.index[0][1]._fields)
        )
        return result

    def get_aggregated_results_as_df(self, *, group: Literal["test", "train"] = "test") -> pd.DataFrame:
        """Get the aggregated results as a pandas DataFrame.

        This will return all `agg__` columns that the scorer returned (see `results_` attribute) as a pandas dataframe.


        The returned Df will have the cv-folds as rows and the aggregated values as columns.
        This makes it convenient to then calculate typical metrics like mean, std, etc. across the cv-folds.

        Returns
        -------
        pd.DataFrame
            The results as a pandas DataFrame.

        """
        result = pd.DataFrame(self.results_)
        relevant_cols = [c for c in result.columns if c.startswith(f"{group}__agg__")]
        result = result[relevant_cols]
        result.columns = [c.split("__", 2)[-1] for c in result.columns]
        result = result.rename_axis(index="fold")
        return result

    def get_raw_results(self, *, group: Literal["test", "train"] = "test") -> dict:
        """Get the raw results of the cross-validation.

        Get the direct output of the algorithms.
        These are usually handed down through the `agg__raw__` parameters of the scoring output.

        The exact structure of the results depends on the scorer and the optimizer used.
        Usually, outputs are provided as pandas dataframes.

        If the individual outputs are dataframes, they are concatenated along the `cv_fold` axis.
        Otherwise, they are simply returned as a list, where each element represents the output of one cv-fold.

        Returns
        -------
        dict
            Raw algorithm results from the cross-validation.

        """
        out = {}
        for k, v in self.results_.items():
            if not k.startswith(f"{group}__single__raw__"):
                continue
            key = k.split("__", 3)[-1]
            if isinstance(v[0], pd.DataFrame):
                out[key] = pd.concat(v, keys=range(len(v)), names=["fold", *v[0].index.names])
            else:
                out[key] = v

        return out


def save_evaluation_results(
    name: str,
    eval_obj: Union[Evaluation[Any], EvaluationCV[Any]],
    *,
    condition: Literal["laboratory", "free_living"],
    base_path: Path,
    raw_results: Union[list[str], bool] = False,
    include_non_stable_results: bool = False,
) -> None:
    """Save the results of an evaluation to a folder.

    This will store, the raw results, the aggregated results and the single results in separate files in a folder
    under the path `base_path / condition / name`.

    Parameters
    ----------
    name
        Name to identify the result (usually the algorithm name).
    eval_obj
        The result object to save.
        Aka the evaluation object after the `run` method has been called.
    condition
        The condition of the evaluation.
        Should be one of "laboratory" or "free-living".
    base_path
        The base path where the results should be stored.
    raw_results
        A list of keys to filter the raw results.
        If provided only these raw results are stored.
        If set to False (default), no raw results are stored.
        If set to True, all raw results are stored.
    include_non_stable_results
        Whether to include non-stable results in the output.
        This means results that might change from run to run even if the algorithm not changed.
        This includes mainly the runtime.
    """
    folder = base_path / condition / name
    folder.mkdir(parents=True, exist_ok=True)
    # Save raw results
    raw_results_vals = eval_obj.get_raw_results()
    if raw_results is False:
        raw_results_vals = {}
    elif raw_results is True:
        pass
    elif isinstance(raw_results, list):
        raw_results_vals = {k: v for k, v in raw_results_vals.items() if k in raw_results}
    else:
        raise ValueError("raw_results must be a list of keys or True or False")
    for k, v in raw_results_vals.items():
        v.to_csv(folder / f"raw_{k}.csv")

    # Save aggregated results
    # Transposing for better readability
    eval_obj.get_aggregated_results_as_df().drop(columns="runtime_s", errors="ignore").T.to_csv(
        folder / "aggregated_results.csv"
    )
    # Save single results
    eval_obj.get_single_results_as_df().drop(columns="runtime_s", errors="ignore").to_csv(folder / "single_results.csv")

    # Save timings
    if include_non_stable_results:
        timing_result = eval_obj.perf_
        with (folder / "timings.json").open("w") as f:
            json.dump(timing_result, f, indent=2)


__all__ = ["Evaluation", "EvaluationCV", "save_evaluation_results"]
