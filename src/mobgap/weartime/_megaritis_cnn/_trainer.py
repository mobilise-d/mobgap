"""Dataset pipeline for training CNN wear-time detectors."""

from __future__ import annotations

import json
from collections.abc import Iterator  # noqa: TC003 - tpcp evaluates annotations at runtime.
from dataclasses import replace
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np  # noqa: TC002 - tpcp evaluates annotations at runtime.
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
from typing_extensions import Self

from mobgap.data.base import BaseGaitDataset
from mobgap.weartime._megaritis_cnn._detector import WtdMegaritisCNN
from mobgap.weartime._megaritis_cnn._model import (
    KerasWeartimeModel,
    _iter_window_batches,
)
from mobgap.weartime._model_config import WearTimeModelConfig  # noqa: TC001 - tpcp resolves method annotations.
from mobgap.weartime._optimization_presets import _sustain_weartime_optimization_defaults
from mobgap.weartime._training import _single_sampling_rate_hz, _TrainingDataFromDataset
from mobgap.weartime.base import TrainingData  # noqa: TC001 - tpcp evaluates annotations at runtime.
from mobgap.weartime.pipeline import WtdEmulationPipeline
from mobgap.weartime.utils.ml_feature_extraction import labels_from_interval_arrays, reference_weartime_interval_arrays


class WtdMegaritisCNNTrainer(WtdEmulationPipeline, OptimizablePipeline[BaseGaitDataset]):
    """Build and fit a CNN on dataset recordings, exposing the fitted detector through ``algo``.

    Training consumes labeled reference regions on their original sample grid. Windows intersecting uncertain
    regions are excluded. Recordings load lazily on demand; raw windows are not cached. The built-in Megaritis CNN
    and CNN-LSTM blueprints serialize per-window standardization inside the network. Supplied native Keras graphs
    receive raw windows and must include their own required preprocessing. The supplied untrained ``algo.model``
    owns its type, graph structure and inputs.
    This trainer only controls fitting; thresholds and prediction batching remain on ``algo.model``.
    ``random_seed`` seeds network initialization, dropout and window shuffling.

    ``PredefinedParameters.empty_cnn`` and ``empty_cnn_lstm`` configure fresh algorithms and initial fitting settings.
    ``PredefinedParameters.sustain_weartime_cnn`` and ``sustain_weartime_cnn_lstm`` restore selected fitting recipes
    without loading model structure or weights.
    ``OptimizationPresets.sustain_weartime`` supplies the search space, grouped inner CV and training subsampling.

    Parameters
    ----------
    algo
        Detector owning an untrained model blueprint. Its concrete model class determines the network type.
    training_overlap
        Training window overlap, independent of inference overlap on algo.model.
    learning_rate
        Adam learning rate.
    gradient_clipnorm
        Optional gradient norm limit passed to Adam. The SUSTAIN CNN-LSTM recipe uses 1.
    batch_size, epochs
        Optimizer batch size and number of passes through the recording dataset.
    shuffle_buffer_size
        Maximum number of raw windows buffered for shuffling. Zero disables shuffling.
    random_seed
        Seed for network initialization, dropout and training-window shuffling.
    fit_verbose
        Keras fit verbosity; does not affect fitted parameters.
    convert_to_body_frame
        Convert each recording to body-frame columns when it is loaded.

    Examples
    --------
    Train a new detector through TPCP and use it independently of the trainer:

    >>> from tpcp.optimize import Optimize
    >>> trained = (
    ...     Optimize(WtdMegaritisCNNTrainer()).optimize(dataset).optimized_pipeline_
    ... )  # doctest: +SKIP
    >>> detector = trained.algo  # doctest: +SKIP
    >>> detector.detect(data, sampling_rate_hz=100)  # doctest: +SKIP
    """

    algo: Parameter[WtdMegaritisCNN]
    algo__model: OptimizableParameter[KerasWeartimeModel]
    algo__model__overlap: PureParameter[float]
    algo__model__prediction_threshold: PureParameter[float]
    algo__model__predict_batch_size: PureParameter[int]
    algo__model__window_batch_size: PureParameter[int]
    algo__waking_hours: PureParameter[tuple]
    training_overlap: HyperParameter[float]
    learning_rate: HyperParameter[float]
    gradient_clipnorm: HyperParameter[Optional[float]]  # noqa: UP045
    batch_size: HyperParameter[int]
    epochs: HyperParameter[int]
    shuffle_buffer_size: HyperParameter[int]
    random_seed: HyperParameter[int]
    convert_to_body_frame: HyperParameter[bool]
    fit_verbose: PureParameter[int]

    class OptimizationPresets:
        """SUSTAIN inner-CV and sampling settings with an initial CNN search space."""

        @staticmethod
        def _create_search_space(trial: Any) -> None:
            trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            trial.suggest_categorical("algo__model__dropout_rate", [0.2, 0.3, 0.5])
            trial.suggest_categorical("batch_size", [256, 512, 1024])

        @classproperty
        def sustain_weartime(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Rank human participant folds with sampled human and simulated training days."""
            return MappingProxyType(
                {"create_search_space": cls._create_search_space, **_sustain_weartime_optimization_defaults()}
            )

    class PredefinedParameters:
        """Complete empty trainer configurations and selected SUSTAIN fitting recipes."""

        @classproperty
        def empty_cnn(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Configure a fresh CNN algorithm and initial fitting settings without packaged artifacts."""
            return MappingProxyType(
                {
                    "algo": WtdMegaritisCNN(**WtdMegaritisCNN.PredefinedParameters.empty_cnn),
                    "training_overlap": 0.75,
                    "learning_rate": 0.001,
                    "gradient_clipnorm": None,
                    "batch_size": 1024,
                    "epochs": 60,
                    "shuffle_buffer_size": 8192,
                    "random_seed": 42,
                    "fit_verbose": 1,
                    "convert_to_body_frame": True,
                }
            )

        @classproperty
        def empty_cnn_lstm(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Configure a fresh CNN-LSTM algorithm with 57 epochs and gradient clipping at 1."""
            return MappingProxyType(
                {
                    **cls.empty_cnn,
                    "algo": WtdMegaritisCNN(**WtdMegaritisCNN.PredefinedParameters.empty_cnn_lstm),
                    "epochs": 57,
                    "gradient_clipnorm": 1,
                }
            )

        @classproperty
        def sustain_weartime_cnn(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Load the CNN fitting recipe without model structure or weights."""
            with (
                files("mobgap.weartime._megaritis_cnn._models")
                .joinpath("cnn_sustain_weartime_training_params.json")
                .open() as file
            ):
                return MappingProxyType(json.load(file))

        @classproperty
        def sustain_weartime_cnn_lstm(cls) -> MappingProxyType[str, Any]:  # noqa: N805
            """Load the CNN-LSTM fitting recipe without model structure or weights."""
            with (
                files("mobgap.weartime._megaritis_cnn._models")
                .joinpath("cnn_lstm_sustain_weartime_training_params.json")
                .open() as file
            ):
                return MappingProxyType(json.load(file))

    def __init__(
        self,
        algo: WtdMegaritisCNN = cf(WtdMegaritisCNN()),
        *,
        training_overlap: float = 0.75,
        learning_rate: float = 0.001,
        gradient_clipnorm: float | None = None,
        batch_size: int = 1024,
        epochs: int = 60,
        shuffle_buffer_size: int = 8192,
        random_seed: int = 42,
        fit_verbose: int = 1,
        convert_to_body_frame: bool = True,
    ) -> None:
        super().__init__(algo, convert_to_body_frame=convert_to_body_frame)
        self.training_overlap = training_overlap
        self.learning_rate = learning_rate
        self.gradient_clipnorm = gradient_clipnorm
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_buffer_size = shuffle_buffer_size
        self.random_seed = random_seed
        self.fit_verbose = fit_verbose

    def save_model(self, directory: str | Path, *, prefix: str) -> None:
        """Persist the configured native model, input settings and fitting recipe.

        Call after optimization, when ``algo.model.model`` holds the fitted Keras network.
        Write ``<prefix>_model.keras``, ``_model_config.json``, ``_model_params.json`` and
        ``_training_params.json`` into ``directory``, creating it when needed. Existing files are overwritten.
        The model definition excludes weights and input configuration; the fitting recipe excludes ``algo``.
        Optuna search results are saved separately by the caller. Use the matching SUSTAIN artifact prefix
        when packaging files for the predefined model and trainer loaders.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model = self.algo.model
        model.model.save(directory / f"{prefix}_model.keras")
        metadata = {
            "model_config": model.model_config.to_dict(),
            "model_params": {
                key: value
                for key, value in model.get_params(deep=False).items()
                if key not in ("model", "model_config")
            },
            "training_params": {key: value for key, value in self.get_params(deep=False).items() if key != "algo"},
        }
        for suffix, values in metadata.items():
            with (directory / f"{prefix}_{suffix}.json").open("w") as file:
                json.dump(values, file, indent=2)
                file.write("\n")

    @make_optimize_safe
    def self_optimize(self, dataset: BaseGaitDataset) -> Self:
        """Fit a new network and store it in a cloned inference detector.

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

        Returns
        -------
        self
            Trainer with the fitted inference detector available as ``algo``.
        """
        tf = import_module("tensorflow")
        sampling_rate_hz = _single_sampling_rate_hz(dataset)
        algo = self.algo.clone()
        if algo.model.model_config.sampling_rate_hz is not None:
            algo.model.model_config.validate_sampling_rate(sampling_rate_hz)
        config = replace(algo.model.model_config, sampling_rate_hz=sampling_rate_hz)
        window_samples, _ = config.window_parameters(self.training_overlap)
        tf.keras.utils.set_random_seed(self.random_seed)
        model = algo.model._create_model(config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.gradient_clipnorm),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        training_data = _TrainingDataFromDataset(dataset, convert_to_body_frame=self.convert_to_body_frame)
        model.fit(
            self._make_tf_dataset(
                training_data, config=config, batch_size=self.batch_size, window_samples=window_samples
            ),
            epochs=self.epochs,
            shuffle=False,
            verbose=self.fit_verbose,
        )
        self.algo = algo.set_params(model__model=model, model__model_config=config)
        return self

    def _iter_training_window_batches(
        self, training_data: TrainingData, config: WearTimeModelConfig
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        window_samples, _ = config.window_parameters(self.training_overlap)
        for data, reference_weartime in training_data:
            reference_interval_arrays = reference_weartime_interval_arrays(reference_weartime)
            for windows, starts in _iter_window_batches(
                data,
                config,
                overlap=self.training_overlap,
                window_batch_size=self.algo.model.window_batch_size,
                reference=reference_weartime,
            ):
                centers = starts + window_samples // 2
                yield windows, labels_from_interval_arrays(centers, *reference_interval_arrays)

    def _make_tf_dataset(
        self,
        training_data: TrainingData,
        *,
        config: WearTimeModelConfig,
        batch_size: int,
        window_samples: int,
    ) -> Any:
        tf = import_module("tensorflow")
        output_signature = (
            tf.TensorSpec(shape=(None, window_samples, len(config.sensor_cols)), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        dataset = tf.data.Dataset.from_generator(
            lambda: self._iter_training_window_batches(training_data, config),
            output_signature=output_signature,
        )
        dataset = dataset.unbatch()
        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(self.shuffle_buffer_size, seed=self.random_seed, reshuffle_each_iteration=True)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
