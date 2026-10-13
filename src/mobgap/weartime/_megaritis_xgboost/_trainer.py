"""Train XGBoost wear-time detectors through TPCP."""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator  # noqa: TC003 - tpcp resolves annotations at runtime.
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Memory  # noqa: TC002 - tpcp resolves annotations at runtime.
from sklearn.base import BaseEstimator
from sklearn.base import clone as clone_untrained_estimator
from sklearn.utils.validation import check_is_fitted
from tpcp import (
    HyperParameter,
    OptimizableParameter,
    OptimizablePipeline,
    Parameter,
    PureParameter,
    cf,
    make_optimize_safe,
)
from tpcp.misc import classproperty
from tpcp.parallel import Parallel, delayed
from typing_extensions import Self, Unpack

from mobgap.data.base import BaseGaitDataset
from mobgap.weartime._megaritis_xgboost._detector import (
    WtdMegaritisXGBoost,
    _FeatureExtractionKwargs,
    _recording_feature_batches,
)
from mobgap.weartime._megaritis_xgboost._features import FEATURE_ORDER_90PCT
from mobgap.weartime._model_config import WearTimeModelConfig  # noqa: TC001 - tpcp evaluates field annotations.
from mobgap.weartime._optimization_presets import _sustain_weartime_optimization_defaults
from mobgap.weartime._training import _single_sampling_rate_hz, _TrainingDataFromDataset
from mobgap.weartime.pipeline import WtdEmulationPipeline
from mobgap.weartime.utils.ml_feature_extraction import (
    known_window_mask,
    labels_from_interval_centers,
    window_start_end,
)


class _TrainingFeatureKwargs(_FeatureExtractionKwargs):
    memory: Memory
    n_jobs_features: int


def _iter_training_recording_feature_batches(
    data: pd.DataFrame,
    reference_weartime: pd.DataFrame,
    *,
    memory: Memory,
    n_jobs_features: int,
    **feature_kwargs: Unpack[_FeatureExtractionKwargs],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    window_start_end_ = window_start_end(len(data), feature_kwargs["window_samples"], feature_kwargs["step_samples"])
    known = known_window_mask(window_start_end_[:, 0], window_start_end_[:, 1], reference_weartime)
    if not known.any():
        return
    feature_batches = _recording_feature_batches(
        data, memory=memory, feature_kwargs=feature_kwargs, n_jobs=n_jobs_features
    )
    reference_centers = window_start_end_[:, 0] + feature_kwargs["window_samples"] // 2
    for batch_index, features in enumerate(feature_batches):
        batch_start = batch_index * feature_kwargs["window_batch_size"]
        labels = labels_from_interval_centers(
            reference_centers[batch_start : batch_start + len(features)], reference_weartime
        )
        batch_known = known[batch_start : batch_start + len(features)]
        if batch_known.all():
            yield features, labels
        elif batch_known.any():
            yield features[batch_known], labels[batch_known]


def _extract_training_data_recording_features(
    training_data: _TrainingDataFromDataset,
    datapoint_index: int,
    **kwargs: Unpack[_TrainingFeatureKwargs],
) -> list[tuple[np.ndarray, np.ndarray]]:
    recording = training_data.load_recording(datapoint_index)
    return list(_iter_training_recording_feature_batches(*recording, **kwargs))


def _iter_training_feature_results(
    training_data: _TrainingDataFromDataset,
    *,
    n_jobs: int,
    **kwargs: Unpack[_TrainingFeatureKwargs],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_jobs != 1:
        recording_batches = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator",
            pre_dispatch="n_jobs",
        )(
            delayed(_extract_training_data_recording_features)(training_data, datapoint_index, **kwargs)
            for datapoint_index in range(len(training_data))
        )
        for recording_batch_results in recording_batches:
            yield from recording_batch_results
    else:
        for recording in training_data:
            yield from _iter_training_recording_feature_batches(*recording, **kwargs)


class WtdMegaritisXGBoostTrainer(WtdEmulationPipeline, OptimizablePipeline[BaseGaitDataset]):
    """Fit an inference detector from a gait dataset and run it through the normal emulation interface.

    The supplied ``algo`` owns its untrained classifier and feature/input configuration. Fitting operates on a clone.
    Recording processes and training overlap belong to this trainer; inference settings remain on the algorithm.

    ``PredefinedParameters.sustain_weartime`` restores selected fitting settings without loading weights.
    ``HyperparameterSearchPresets.sustain_weartime`` supplies the search space,
    grouped inner CV and training subsampling.

    Parameters
    ----------
    algo
        Untrained detector defining the classifier type and model inputs. Its clf is fitted on a cloned detector.
    training_overlap
        Fractional training window overlap, independent of algo.overlap during inference.
    n_jobs
        Joblib processes over training recordings. algo.n_jobs_features controls feature threads per recording.
    convert_to_body_frame
        Convert recording columns to the body frame lazily while loading training data.

    Examples
    --------
    >>> from tpcp.optimize import Optimize
    >>> trained = (
    ...     Optimize(WtdMegaritisXGBoostTrainer()).optimize(dataset).optimized_pipeline_
    ... )  # doctest: +SKIP
    >>> detector = trained.algo  # doctest: +SKIP
    >>> detector.detect(data, sampling_rate_hz=100)  # doctest: +SKIP
    """

    algo: Parameter[WtdMegaritisXGBoost]
    algo__clf: OptimizableParameter[Optional[BaseEstimator]]  # noqa: UP045
    algo__model_config: OptimizableParameter[Optional[WearTimeModelConfig]]  # noqa: UP045
    algo__overlap: PureParameter[float]
    algo__prediction_threshold: PureParameter[float]
    algo__window_batch_size: PureParameter[int]
    algo__n_jobs_features: PureParameter[int]
    algo__memory: PureParameter[Memory]
    algo__waking_hours: PureParameter[tuple]
    training_overlap: HyperParameter[float]
    convert_to_body_frame: HyperParameter[bool]
    n_jobs: PureParameter[int]

    class PredefinedParameters:
        """Templates for new training and the selected settings of a packaged SUSTAIN run."""

        @classproperty
        def sustain_weartime(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Load fitting settings without constructing or loading a model."""
            with (
                files("mobgap.weartime._megaritis_xgboost._models")
                .joinpath("xgboost_sustain_weartime_training_params.json")
                .open() as file
            ):
                return MappingProxyType(json.load(file))

    class HyperparameterSearchPresets:
        """SUSTAIN-specific inner search settings; sampling and CV belong to the search layer."""

        @staticmethod
        def _create_search_space(trial: Any) -> None:
            for name, choices in {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.8],
                "min_child_weight": [1, 3],
            }.items():
                trial.suggest_categorical(f"algo__clf__{name}", choices)

        @classproperty
        def sustain_weartime(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Rank three human participant folds, with sampled human and simulated training days."""
            return MappingProxyType(
                {"create_search_space": cls._create_search_space, **_sustain_weartime_optimization_defaults()}
            )

    def __init__(
        self,
        algo: WtdMegaritisXGBoost = cf(WtdMegaritisXGBoost()),
        *,
        training_overlap: float = 0.75,
        n_jobs: int = 1,
        convert_to_body_frame: bool = True,
    ) -> None:
        super().__init__(algo, convert_to_body_frame=convert_to_body_frame)
        self.training_overlap = training_overlap
        self.n_jobs = n_jobs

    def save_model(self, directory: str | Path, *, prefix: str) -> None:
        """Persist the fitted estimator, its untrained template, input settings and fitting recipe.

        Call after optimization. Write ``<prefix>_model.pkl``, ``_classifier_template.pkl``,
        ``_model_config.json`` and ``_training_params.json`` into ``directory``, creating it when needed.
        Existing files are overwritten. The template retains constructor parameters with learned state
        removed using sklearn cloning; the fitting recipe excludes ``algo``. Optuna search results are
        saved separately by the caller. Use the matching SUSTAIN artifact prefix when packaging files for
        the predefined model and trainer loaders.
        """
        check_is_fitted(self.algo.clf)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{prefix}_model.pkl").open("wb") as file:
            pickle.dump(self.algo.clf, file)
        with (directory / f"{prefix}_classifier_template.pkl").open("wb") as file:
            pickle.dump(clone_untrained_estimator(self.algo.clf), file)
        metadata = {
            "model_config": self.algo.model_config.to_dict(),
            "training_params": {key: value for key, value in self.get_params(deep=False).items() if key != "algo"},
        }
        for suffix, values in metadata.items():
            with (directory / f"{prefix}_{suffix}.json").open("w") as file:
                json.dump(values, file, indent=2)
                file.write("\n")

    @make_optimize_safe
    def self_optimize(self, dataset: BaseGaitDataset, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Clone and fit the classifier template, retaining the detector's inference configuration.

        Parameters
        ----------
        dataset
            Dataset whose individual datapoints expose ``data_ss`` (one sensor DataFrame), ``sampling_rate_hz``
            and ``reference_weartime_``. All datapoints must share a common sampling rate. Reference rows are
            half-open ``[start, end)`` sample intervals on the original recording grid, with integer ``start``
            and ``end`` columns and a mandatory unordered categorical ``label`` column with categories
            ``["wear", "uncertain"]``, including empty/all-wear reference frames. ``uncertain`` means wear versus
            non-wear could not be determined; the complement of labeled intervals is known non-wear.
            Entire windows intersecting uncertain regions are excluded from fitting. Recordings are loaded lazily.

        **kwargs
            Additional keyword arguments passed to the cloned classifier's ``fit`` method.

        Returns
        -------
        self
            Trainer with the fitted inference detector available as ``algo``.
        """
        sampling_rate_hz = _single_sampling_rate_hz(dataset)
        algo = self.algo.clone()
        if algo.model_config.sampling_rate_hz is not None:
            algo.model_config.validate_sampling_rate(sampling_rate_hz)
        config = replace(algo.model_config, sampling_rate_hz=sampling_rate_hz)
        window_samples, step_samples = config.window_parameters(self.training_overlap)
        classifier = clone_untrained_estimator(algo.clf) if algo.clf is not None else algo._create_classifier()
        training_data = _TrainingDataFromDataset(dataset, convert_to_body_frame=self.convert_to_body_frame)
        feature_batches = []
        label_batches = []
        for features, labels in _iter_training_feature_results(
            training_data,
            memory=self.algo.memory,
            n_jobs=self.n_jobs,
            n_jobs_features=self.algo.n_jobs_features,
            sampling_rate_hz=sampling_rate_hz,
            window_samples=window_samples,
            step_samples=step_samples,
            window_batch_size=self.algo.window_batch_size,
            version="lightweight" if set(config.feature_names).issubset(FEATURE_ORDER_90PCT) else "full",
            sensor_cols=config.sensor_cols,
            feature_names=config.feature_names,
        ):
            feature_batches.append(features)
            label_batches.append(labels)
        if not feature_batches:
            raise ValueError("The training data did not yield any XGBoost feature windows.")
        feature_values = feature_batches[0] if len(feature_batches) == 1 else np.concatenate(feature_batches)
        labels = label_batches[0] if len(label_batches) == 1 else np.concatenate(label_batches)
        classifier.fit(pd.DataFrame(feature_values, columns=config.feature_names, copy=False), labels, **kwargs)
        self.algo = algo.set_params(clf=classifier, model_config=config)
        return self
