"""Tests for the Megaritis CNN wear-time detector."""

from __future__ import annotations

import subprocess
import sys
from datetime import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.utils.array_handling import bool_array_to_start_end_array
from mobgap.weartime import (
    KerasWeartimeModel,
    MegaritisCnnLstmWeartimeModel,
    MegaritisCnnWeartimeModel,
    WearTimeModelConfig,
    WtdMegaritisCNN,
    WtdMegaritisCNNTrainer,
    load_keras_weartime_model,
)
from mobgap.weartime import _keras_weartime_model as keras_model_module
from mobgap.weartime import _wtd_megaritis_cnn as cnn_module
from mobgap.weartime import _wtd_megaritis_cnn_trainer as trainer_module
from mobgap.weartime.utils.windows_to_weartime import (
    filter_short_wear_bouts_by_confidence,
    overlapping_window_predictions_to_sample_labels,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FixedPredictionModel:
    """Small test double that returns fixed per-window wear probabilities."""

    def __init__(self, predictions: list[int]) -> None:
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, x_batch: np.ndarray, *, verbose: int = 0, batch_size: int = 256) -> np.ndarray:
        if verbose != 0:
            raise AssertionError("The CNN detector should call predict with verbose=0.")
        if batch_size != 256:
            raise AssertionError("Unexpected CNN prediction batch size.")
        if len(x_batch) != len(self.predictions):
            raise AssertionError(f"Expected {len(self.predictions)} windows, got {len(x_batch)}.")
        return self.predictions.reshape(-1, 1)


def _keras_window_model(
    predictions: list[int], *, window_sec: float = 20.0, overlap: float = 0.0
) -> KerasWeartimeModel:
    return KerasWeartimeModel(
        model=_FixedPredictionModel(predictions),
        model_config=WearTimeModelConfig(1.0, window_sec),
        overlap=overlap,
    )


def _sensor_data(n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(np.ones((n_samples, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)


def _weartime_list(intervals: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(intervals, columns=["start", "end"]).rename_axis(index="wt_id").astype("int64")


class TestMetaWtdMegaritisCNN(TestAlgorithmMixin):
    """Test tpcp algorithm compatibility."""

    __test__ = True

    ALGORITHM_CLASS = WtdMegaritisCNN

    @pytest.fixture
    def after_action_instance(self) -> WtdMegaritisCNN:
        """Create a detector after action for the algorithm mixin."""
        return self.ALGORITHM_CLASS(
            model=_keras_window_model([0, 0, 0, 0, 0, 0]), waking_hours=(time(0), time(0, 2))
        ).detect(
            _sensor_data(120),
            sampling_rate_hz=1.0,
        )


class TestMetaKerasWeartimeModel(TestAlgorithmMixin):
    """Test tpcp algorithm compatibility for the low-level Keras model."""

    __test__ = True

    ALGORITHM_CLASS = KerasWeartimeModel

    @pytest.fixture
    def after_action_instance(self) -> KerasWeartimeModel:
        """Create a low-level model after action for the algorithm mixin."""
        return _keras_window_model([0, 1, 0], window_sec=20.0, overlap=0.0).run(
            _sensor_data(60),
            sampling_rate_hz=1.0,
        )


class TestKerasWeartimeModel:
    """Test the low-level Keras model wrapper behavior."""

    def test_run_creates_window_predictions_without_tensorflow_import(self) -> None:
        """Classify recording windows using a preloaded model object."""
        result = _keras_window_model([0, 1, 1], window_sec=20.0, overlap=0.0).run(
            _sensor_data(60),
            sampling_rate_hz=1.0,
        )

        assert_array_equal(result.window_predictions_, np.array([0, 1, 1], dtype=np.int32))
        assert_array_equal(result.window_start_end_, np.array([[0, 20], [20, 40], [40, 60]], dtype=np.int64))
        assert result.window_samples_ == 20
        assert result.step_samples_ == 20

    def test_probability_at_threshold_counts_as_wear(self) -> None:
        """Use the same inclusive probability threshold as the tree detector."""
        result = _keras_window_model([0.5]).run(_sensor_data(20), sampling_rate_hz=1)
        assert_array_equal(result.window_predictions_, [1])

    def test_run_rejects_sampling_rate_mismatch(self) -> None:
        """Reject inference at a sampling rate that differs from the loaded model."""
        model = KerasWeartimeModel(
            model=_FixedPredictionModel([0]),
            model_config=WearTimeModelConfig(100.0, 20),
            overlap=0.0,
        )

        with pytest.raises(ValueError, match=r"requires 100\.0 Hz"):
            model.run(_sensor_data(60), sampling_rate_hz=1.0)

    @pytest.mark.parametrize("overlap", [-0.1, 1.0])
    def test_run_rejects_invalid_overlap(self, overlap: float) -> None:
        """Reject window strides that leave gaps or cannot advance."""
        with pytest.raises(ValueError, match="overlap"):
            _keras_window_model([0, 0, 0], overlap=overlap).run(_sensor_data(60), sampling_rate_hz=1.0)

    def test_model_standardization_layer_matches_numpy_and_serializes(self, tmp_path: Path) -> None:
        """Run model-side standardization through Keras and keep it loadable."""
        tf = pytest.importorskip("tensorflow")
        layer_cls = keras_model_module._get_per_window_standardization_layer_class()
        rng = np.random.default_rng(42)
        windows = rng.normal(loc=9.81, scale=0.2, size=(8, 20, len(BF_SENSOR_COLS))).astype(np.float32)
        windows[0] = 3.0

        result = layer_cls()(tf.constant(windows)).numpy()
        centered = windows - windows.mean(axis=1, keepdims=True)
        expected = centered / np.sqrt(np.maximum(np.mean(centered**2, axis=1, keepdims=True), 1e-16))

        assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)

        model = MegaritisCnnWeartimeModel()._create_model(WearTimeModelConfig(1, 20))
        model_path = tmp_path / "model.keras"
        model.save(model_path)
        loaded_model = load_keras_weartime_model(model_path)

        assert np.allclose(
            loaded_model(windows, training=False).numpy(), model(windows, training=False).numpy(), rtol=1e-6
        )

    def test_model_with_standardization_loads_in_fresh_process(self, tmp_path: Path) -> None:
        """A saved model preserves predictions when loaded without prior layer registration."""
        pytest.importorskip("tensorflow")
        model = MegaritisCnnWeartimeModel()._create_model(WearTimeModelConfig(1, 20))
        model_path = tmp_path / "trained.keras"
        model.save(model_path)

        windows = np.random.default_rng(42).normal(size=(2, 20, len(BF_SENSOR_COLS))).astype(np.float32)
        np.save(tmp_path / "windows.npy", windows)
        np.save(tmp_path / "expected.npy", model(windows, training=False).numpy())
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """import sys
from pathlib import Path
import numpy as np
from mobgap.weartime import load_keras_weartime_model
model = load_keras_weartime_model(sys.argv[1])
root = Path(sys.argv[1]).parent
actual = model(np.load(root / 'windows.npy'), training=False).numpy()
np.testing.assert_allclose(actual, np.load(root / 'expected.npy'), rtol=1e-6)
""",
                str(model_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr

    def test_no_window_predictions_leave_short_recording_nonwear(self) -> None:
        """Short recordings without a classified window remain nonwear."""
        labels, vote_counts = overlapping_window_predictions_to_sample_labels([], 10, 500, 125)

        assert_array_equal(labels, np.zeros(10, dtype=np.int32))
        assert_array_equal(vote_counts, np.zeros((10, 2), dtype=np.int32))

    def test_window_batches_preserve_raw_sensor_values(self) -> None:
        """Yield raw windows when the Keras model owns standardization."""
        data = pd.DataFrame(
            np.arange(60 * len(BF_SENSOR_COLS), dtype=np.float32).reshape(60, len(BF_SENSOR_COLS)),
            columns=BF_SENSOR_COLS,
        )

        result = list(
            (
                windows
                for windows, _ in keras_model_module._iter_window_batches(
                    data, WearTimeModelConfig(1, 20), overlap=0, window_batch_size=1024
                )
            )
        )

        assert len(result) == 1
        assert_array_equal(result[0], data.to_numpy(dtype=np.float32, copy=False).reshape(3, 20, len(BF_SENSOR_COLS)))


class TestWtdMegaritisCNN:
    """Test the CNN detector behavior with deterministic model outputs."""

    @pytest.mark.parametrize(
        ("model_type", "prefix", "preset_name"),
        [
            (MegaritisCnnWeartimeModel, "cnn_sustain_weartime", "sustain_weartime_cnn"),
            (MegaritisCnnLstmWeartimeModel, "cnn_lstm_sustain_weartime", "sustain_weartime_cnn_lstm"),
        ],
    )
    def test_sustain_presets_separate_model_and_training_settings(
        self, model_type, prefix, preset_name, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load each fitted model and its untrained blueprint independently of the fitting recipe."""
        pytest.importorskip("tensorflow")
        config = WearTimeModelConfig(1, 20)
        blueprint = model_type(dropout_rate=0.5, filters=(2, 2, 2), dense_units=2)
        model = blueprint._create_model(config)
        recipe = {"learning_rate": 0.007, "batch_size": 512, "gradient_clipnorm": 1, "epochs": 7}
        writer = WtdMegaritisCNNTrainer(
            WtdMegaritisCNN(model=blueprint.set_params(model=model, model_config=config)), **recipe
        )
        writer.save_model(tmp_path, prefix=prefix)
        monkeypatch.setattr(keras_model_module, "files", lambda _: tmp_path)
        monkeypatch.setattr(cnn_module, "files", lambda _: tmp_path)
        monkeypatch.setattr(trainer_module, "files", lambda _: tmp_path)
        keras_model_module._load_keras_model_resource.cache_clear()
        parameters = getattr(WtdMegaritisCNN.PredefinedParameters, preset_name)
        fitted = WtdMegaritisCNN(**parameters).model
        untrained = WtdMegaritisCNN(**getattr(WtdMegaritisCNN.PredefinedParameters, "untrained_" + preset_name)).model
        trainer_params = getattr(WtdMegaritisCNNTrainer.PredefinedParameters, preset_name)
        restored_trainer = WtdMegaritisCNNTrainer(WtdMegaritisCNN(model=untrained), **trainer_params)
        assert restored_trainer.learning_rate == 0.007
        assert restored_trainer.batch_size == 512
        assert restored_trainer.gradient_clipnorm == 1
        assert restored_trainer.epochs == 7
        assert type(fitted) is model_type
        assert type(untrained) is model_type
        assert untrained.model is None
        assert untrained.dropout_rate == 0.5
        assert fitted.model_config == config
        windows = np.random.default_rng(42).normal(size=(2, 20, 6)).astype(np.float32)
        assert np.allclose(fitted.model(windows, training=False).numpy(), model(windows, training=False).numpy())
        keras_model_module._load_keras_model_resource.cache_clear()

    def test_fixed_window_predictions_regression(self) -> None:
        """Convert deterministic window predictions to expected wear-time intervals."""
        result = WtdMegaritisCNN(
            model=_keras_window_model([1, 1, 0, 0, 1, 1]),
            waking_hours=(time(0), time(0, 2)),
        ).detect(
            _sensor_data(120),
            sampling_rate_hz=1.0,
        )

        assert_frame_equal(result.weartime_list_, _weartime_list([(0, 40), (80, 120)]))
        assert result.total_weartime_samples_ == 80
        assert result.total_weartime_min_ == pytest.approx(80 / 60)
        assert result.total_weartime_during_waking_min_ == pytest.approx(80 / 60)

    def test_localized_signal_adds_interval_datetimes(self) -> None:
        data = _sensor_data(120)
        data.index = pd.date_range("2026-03-29 12:00", periods=120, freq="s", tz="Europe/London")
        result = WtdMegaritisCNN(model=_keras_window_model([1, 1, 0, 0, 1, 1])).detect(data, sampling_rate_hz=1.0)

        assert result.weartime_list_.local_datetime_start.iloc[0] == data.index[0]
        assert result.weartime_list_.local_datetime_end.iloc[-1] == data.index[-1] + pd.Timedelta(seconds=1)

    def test_short_ambiguous_interior_wear_bout_is_filtered(self) -> None:
        """Remove short interior wear bouts with insufficient wear-vote confidence."""
        sample_labels = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32)
        vote_counts = np.ones((len(sample_labels), 2), dtype=np.int32)

        result = filter_short_wear_bouts_by_confidence(
            wear_intervals=bool_array_to_start_end_array(sample_labels),
            vote_counts=vote_counts,
            data_length=len(sample_labels),
            sampling_rate_hz=1.0,
            min_confidence_short_bouts=0.90,
            short_bout_threshold_minutes=20,
            min_bout_duration_seconds=3,
        )

        assert_array_equal(result, np.empty((0, 2), dtype=np.int64))
