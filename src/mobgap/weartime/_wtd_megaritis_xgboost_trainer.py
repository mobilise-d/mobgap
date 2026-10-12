"""Train XGBoost wear-time detectors through TPCP."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence  # noqa: TC003 - tpcp resolves annotations at runtime.
from importlib import import_module
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Memory  # noqa: TC002 - tpcp resolves annotations at runtime.
from sklearn.base import BaseEstimator  # noqa: TC002 - tpcp resolves annotations at runtime.
from tpcp import (
    HyperParameter,
    OptimizableParameter,
    OptimizablePipeline,
    Parameter,
    PureParameter,
    cf,
    clone,
    make_optimize_safe,
)
from tpcp.misc import classproperty
from tpcp.parallel import Parallel, delayed
from typing_extensions import Self, Unpack

from mobgap.consts import BF_SENSOR_COLS
from mobgap.data.base import BaseGaitDataset
from mobgap.weartime._model_config import WearTimeModelConfig
from mobgap.weartime._optimization_presets import _sustain_weartime_optimization_defaults
from mobgap.weartime._training import _single_sampling_rate_hz, _TrainingDataFromDataset
from mobgap.weartime._wtd_megaritis_xgboost import (
    WtdMegaritisXGBoost,
    _FeatureExtractionKwargs,
    _recording_feature_batches,
)
from mobgap.weartime.pipeline import WtdEmulationPipeline
from mobgap.weartime.utils.feature_extraction import FEATURE_ORDER_90PCT
from mobgap.weartime.utils.ml_feature_extraction import (
    known_window_mask,
    labels_from_interval_centers,
    window_start_end,
)


class _TrainingFeatureKwargs(_FeatureExtractionKwargs):
    memory: Memory
    n_jobs_features: int


def _new_xgboost_classifier() -> BaseEstimator:
    try:
        xgboost = import_module("xgboost")
    except ImportError as exc:
        raise ImportError(
            "The optional dependency `xgboost` is required to train an XGBoost wear-time model. "
            "Install MobGap with the `weartime` extra to use `WtdMegaritisXGBoostTrainer`."
        ) from exc
    return xgboost.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
    )


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

    ``classifier`` is an untrained sklearn-compatible template; fitting uses a clone. ``algo`` holds the fitted
    classifier and immutable input configuration after optimization. Recording processes and training overlap are
    independent of the detector's feature threads and inference overlap. To search classifier parameters, supply
    a configured template, for example ``classifier=XGBClassifier(...)``.

    ``PredefinedParameters.sustain_weartime`` restores selected fitting settings without loading weights.
    ``OptimizationPresets.sustain_weartime`` supplies the search space, grouped inner CV and training subsampling.

    Parameters
    ----------
    algo
        Inference detector to fill with a fitted classifier and input configuration.
    classifier
        Untrained sklearn classifier template. A clone is fitted; None creates the baseline XGBoost classifier.
        Configure the classifier's n_jobs to control its native training and prediction threads.
    window_sec, sensor_cols, feature_names
        Trained window duration, ordered sensor channels and ordered feature columns.
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
    classifier: HyperParameter[Optional[BaseEstimator]]  # noqa: UP045
    window_sec: HyperParameter[float]
    sensor_cols: HyperParameter[Sequence[str]]
    feature_names: HyperParameter[Sequence[str]]
    training_overlap: HyperParameter[float]
    convert_to_body_frame: HyperParameter[bool]
    n_jobs: PureParameter[int]

    class PredefinedParameters:
        """Templates for new training and the selected settings of a packaged SUSTAIN run."""

        @classproperty
        def untrained_sustain_weartime(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Create the baseline classifier template without loading fitted weights."""
            return MappingProxyType({"classifier": _new_xgboost_classifier()})

        @classproperty
        def sustain_weartime(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Restore selected classifier parameters and trained input requirements for retraining."""
            resources = files("mobgap.weartime.production_models")
            with resources.joinpath("xgboost_sustain_weartime_best_params.json").open() as file:
                parameters = json.load(file)
            with resources.joinpath("xgboost_sustain_weartime_model_config.json").open() as file:
                config = WearTimeModelConfig.from_dict(json.load(file))
            classifier = _new_xgboost_classifier().set_params(
                **{name.removeprefix("classifier__"): value for name, value in parameters.items()}
            )
            return MappingProxyType(
                {
                    "classifier": classifier,
                    "window_sec": config.window_sec,
                    "sensor_cols": config.sensor_cols,
                    "feature_names": config.feature_names,
                }
            )

    class OptimizationPresets:
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
                trial.suggest_categorical(f"classifier__{name}", choices)

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
        classifier: Optional[BaseEstimator] = None,  # noqa: UP045
        window_sec: float = 5.0,
        sensor_cols: Sequence[str] = tuple(BF_SENSOR_COLS),
        feature_names: Sequence[str] = tuple(FEATURE_ORDER_90PCT),
        training_overlap: float = 0.75,
        n_jobs: int = 1,
        convert_to_body_frame: bool = True,
    ) -> None:
        super().__init__(algo, convert_to_body_frame=convert_to_body_frame)
        self.classifier = classifier
        self.window_sec = window_sec
        self.sensor_cols = sensor_cols
        self.feature_names = feature_names
        self.training_overlap = training_overlap
        self.n_jobs = n_jobs

    @make_optimize_safe
    def self_optimize(self, dataset: BaseGaitDataset, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Clone and fit the classifier template, retaining the detector's inference configuration."""
        sampling_rate_hz = _single_sampling_rate_hz(dataset)
        config = WearTimeModelConfig(
            sampling_rate_hz, self.window_sec, tuple(self.sensor_cols), tuple(self.feature_names)
        )
        window_samples, step_samples = config.window_parameters(self.training_overlap)
        classifier = clone(self.classifier) if self.classifier is not None else _new_xgboost_classifier()
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
        self.algo = self.algo.clone().set_params(clf=classifier, model_config=config)
        return self
