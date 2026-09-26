"""Tests for the Megaritis CNN wear-time detector."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.weartime import MegaritisCnnLstmWeartimeModel, MegaritisCnnWeartimeModel, WtdMegaritisCNN
from mobgap.weartime import _keras_weartime_model as keras_model_module
from mobgap.weartime.utils._intervals import flags_to_intervals
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


class _SelfOptimizeRecorder(MegaritisCnnWeartimeModel):
    """Keras model double that records delegated self-optimization calls."""

    def __init__(self, returned_model: MegaritisCnnWeartimeModel | None = None) -> None:
        super().__init__()
        self.returned_model = returned_model
        self.optimize_training_data = None
        self.optimize_sampling_rate_hz = None
        self.optimize_recording_sample_counts = None

    def self_optimize(
        self,
        training_data: list[tuple[pd.DataFrame, pd.DataFrame]],
        *,
        sampling_rate_hz: float,
        recording_sample_counts: tuple[int, ...],
    ) -> MegaritisCnnWeartimeModel:
        self.optimize_training_data = training_data
        self.optimize_sampling_rate_hz = sampling_rate_hz
        self.optimize_recording_sample_counts = recording_sample_counts
        return self.returned_model or self


def _keras_window_model(
    predictions: list[int], *, window_sec: float = 20.0, overlap: float = 0.0
) -> MegaritisCnnLstmWeartimeModel:
    return MegaritisCnnLstmWeartimeModel(
        _model=_FixedPredictionModel(predictions),
        _trained_sampling_rate_hz=1.0,
        window_sec=window_sec,
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
        return self.ALGORITHM_CLASS(model=_keras_window_model([0, 0, 0, 0, 0, 0]), waking_hours_min=(0, 2)).detect(
            _sensor_data(120),
            sampling_rate_hz=1.0,
        )


class TestMetaMegaritisCnnLstmWeartimeModel(TestAlgorithmMixin):
    """Test tpcp algorithm compatibility for the low-level Keras model."""

    __test__ = True

    ALGORITHM_CLASS = MegaritisCnnLstmWeartimeModel

    @pytest.fixture
    def after_action_instance(self) -> MegaritisCnnLstmWeartimeModel:
        """Create a low-level model after action for the algorithm mixin."""
        return _keras_window_model([0, 1, 0], window_sec=20.0, overlap=0.0).run(
            _sensor_data(60),
            sampling_rate_hz=1.0,
        )


class TestMegaritisCnnWeartimeModel:
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

    def test_run_rejects_sampling_rate_mismatch(self) -> None:
        """Reject inference at a sampling rate that differs from the loaded model."""
        model = MegaritisCnnWeartimeModel(
            _model=_FixedPredictionModel([0]),
            _trained_sampling_rate_hz=100.0,
            window_sec=20.0,
            overlap=0.0,
        )

        with pytest.raises(ValueError, match=r"trained for 100\.0 Hz"):
            model.run(_sensor_data(60), sampling_rate_hz=1.0)

    @pytest.mark.parametrize("overlap", [-0.1, 1.0])
    def test_run_rejects_invalid_overlap(self, overlap: float) -> None:
        """Reject window strides that leave gaps or cannot advance."""
        with pytest.raises(ValueError, match="overlap"):
            _keras_window_model([0, 0, 0], overlap=overlap).run(_sensor_data(60), sampling_rate_hz=1.0)

    def test_self_optimize_rejects_already_loaded_model(self) -> None:
        """Do not train over an already loaded model."""
        model = MegaritisCnnWeartimeModel(_model=_FixedPredictionModel([0]), _trained_sampling_rate_hz=1.0)

        with pytest.raises(RuntimeError, match="already trained or loaded"):
            model.self_optimize([], sampling_rate_hz=1.0, recording_sample_counts=())

    def test_self_optimize_rejects_one_shot_iterators_for_multi_epoch_training(self) -> None:
        """Require re-iterable lazy training data for multi-epoch fitting."""
        training_data = iter([(_sensor_data(60), _weartime_list([(0, 60)]))])

        with pytest.raises(ValueError, match="re-iterable"):
            MegaritisCnnWeartimeModel(epochs=2).self_optimize(
                training_data,
                sampling_rate_hz=1.0,
                recording_sample_counts=(60,),
            )

    def test_steps_per_epoch_uses_per_recording_sample_counts(self) -> None:
        """Calculate epoch length without treating separate recordings as one continuous signal."""
        total_windows, steps_per_epoch = keras_model_module._steps_per_epoch_from_recording_sample_counts(
            (25, 35),
            window_samples=10,
            step_samples=10,
            batch_size=4,
        )

        assert total_windows == 5
        assert steps_per_epoch == 2

    def test_standardize_windows_matches_mean_std_formula(self) -> None:
        """Standardize windows without changing the numerical result."""
        rng = np.random.default_rng(42)
        windows = rng.normal(loc=9.81, scale=0.2, size=(8, 20, len(BF_SENSOR_COLS))).astype(np.float32)
        windows[0] = 3.0
        original_windows = windows.copy()

        expected = original_windows - original_windows.mean(axis=1, keepdims=True)
        expected_std = original_windows.std(axis=1, keepdims=True)
        expected_std[expected_std < 1e-8] = 1e-8
        expected = (expected / expected_std).astype(np.float32, copy=False)

        result = keras_model_module._standardize_windows(windows)

        assert result.dtype == np.float32
        assert np.allclose(result, expected)
        assert_array_equal(result[0], np.zeros_like(result[0]))

    def test_model_input_sensor_array_is_float32_and_contiguous(self) -> None:
        """Prepare recording-level sensor arrays once before batching windows."""
        source = np.arange(20 * len(BF_SENSOR_COLS) * 2, dtype=np.float64).reshape(20, len(BF_SENSOR_COLS) * 2)
        sensor_view = source[:, ::2]

        result = keras_model_module._as_model_input_sensor_array(sensor_view)

        assert result.dtype == np.float32
        assert result.flags.c_contiguous
        assert_array_equal(result, sensor_view.astype(np.float32))

        already_prepared = np.ascontiguousarray(sensor_view.astype(np.float32))
        assert keras_model_module._as_model_input_sensor_array(already_prepared) is already_prepared

    def test_model_standardization_layer_matches_numpy_and_serializes(self, tmp_path: Path) -> None:
        """Run model-side standardization through Keras and keep it loadable."""
        tf = pytest.importorskip("tensorflow")
        keras_models = pytest.importorskip("tensorflow.keras.models")
        layer_cls = keras_model_module._get_per_window_standardization_layer_class()
        rng = np.random.default_rng(42)
        windows = rng.normal(loc=9.81, scale=0.2, size=(8, 20, len(BF_SENSOR_COLS))).astype(np.float32)

        result = layer_cls()(tf.constant(windows)).numpy()
        expected = keras_model_module._standardize_windows(windows.copy())

        assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)

        model = MegaritisCnnWeartimeModel(standardize_in_model=True)._create_model((20, len(BF_SENSOR_COLS)))
        model_path = tmp_path / "model.keras"
        model.save(model_path)
        loaded_model = keras_models.load_model(model_path)

        assert any(layer.name == "per_window_standardization" for layer in loaded_model.layers)

    def test_model_with_standardization_loads_in_fresh_process(self, tmp_path: Path) -> None:
        """A saved training artifact can be loaded without prior layer registration."""
        pytest.importorskip("tensorflow")
        model = MegaritisCnnWeartimeModel(standardize_in_model=True)._create_model((20, len(BF_SENSOR_COLS)))
        model_path = tmp_path / "trained.keras"
        model.save(model_path)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from mobgap.weartime import load_keras_weartime_model; "
                "model = load_keras_weartime_model(sys.argv[1]); "
                "assert any(layer.name == 'per_window_standardization' for layer in model.layers)",
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

    def test_stepped_window_view_matches_advanced_indexing(self) -> None:
        """Create the same stepped windows for C- and Fortran-ordered arrays."""
        sensor_data = np.arange(10 * len(BF_SENSOR_COLS), dtype=np.float32).reshape(10, len(BF_SENSOR_COLS))
        starts = np.arange(0, 10 - 4 + 1, 2)
        offsets = np.arange(4)

        for array in [sensor_data, np.asfortranarray(sensor_data)]:
            result = keras_model_module._stepped_window_view(array, window_samples=4, step_samples=2, n_windows=4)
            expected = array[starts[:, None] + offsets]
            assert_array_equal(result, expected)
            assert not result.flags.writeable

    def test_non_overlapping_stepped_window_view_is_contiguous(self) -> None:
        """Use a contiguous reshape view for non-overlapping windows."""
        sensor_data = np.arange(12 * len(BF_SENSOR_COLS), dtype=np.float32).reshape(12, len(BF_SENSOR_COLS))

        result = keras_model_module._stepped_window_view(
            sensor_data,
            window_samples=4,
            step_samples=4,
            n_windows=3,
        )

        assert_array_equal(result, sensor_data.reshape(3, 4, len(BF_SENSOR_COLS)))
        assert result.flags.c_contiguous
        assert not result.flags.writeable
        assert np.shares_memory(result, sensor_data)

    def test_model_standardization_skips_cpu_standardization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Yield raw windows when the Keras model owns standardization."""
        data = pd.DataFrame(
            np.arange(60 * len(BF_SENSOR_COLS), dtype=np.float32).reshape(60, len(BF_SENSOR_COLS)),
            columns=BF_SENSOR_COLS,
        )

        def _raise_if_called(_windows: np.ndarray) -> np.ndarray:
            raise AssertionError("CPU standardization should be skipped.")

        monkeypatch.setattr(keras_model_module, "_standardize_windows", _raise_if_called)

        result = list(
            MegaritisCnnWeartimeModel(standardize_in_model=True, window_sec=20.0, overlap=0.0)._iter_window_batches(
                data,
                sampling_rate_hz=1.0,
            )
        )

        assert len(result) == 1
        assert_array_equal(
            result[0][0], data.to_numpy(dtype=np.float32, copy=False).reshape(3, 20, len(BF_SENSOR_COLS))
        )

    def test_non_overlapping_model_standardization_reuses_window_view(self) -> None:
        """Avoid an extra batch copy for contiguous non-overlapping windows."""
        data = pd.DataFrame(
            np.arange(60 * len(BF_SENSOR_COLS), dtype=np.float32).reshape(60, len(BF_SENSOR_COLS)),
            columns=BF_SENSOR_COLS,
        )

        result = list(
            MegaritisCnnWeartimeModel(standardize_in_model=True, window_sec=20.0, overlap=0.0)._iter_window_batches(
                data,
                sampling_rate_hz=1.0,
            )
        )

        windows = result[0][0]
        assert_array_equal(windows, data.to_numpy(dtype=np.float32, copy=False).reshape(3, 20, len(BF_SENSOR_COLS)))
        assert windows.flags.c_contiguous
        assert not windows.flags.writeable
        assert not windows.flags.owndata

    def test_training_dataset_repeats_after_batching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep Keras multi-epoch training from exhausting the dataset between epochs."""

        class _FakeDataset:
            last: _FakeDataset | None = None

            def __init__(self) -> None:
                self.operations: list[object] = []

            @classmethod
            def from_generator(cls, generator_factory: object, *, output_signature: object) -> _FakeDataset:
                dataset = cls()
                dataset.operations.append(("from_generator", generator_factory, output_signature))
                cls.last = dataset
                return dataset

            def unbatch(self) -> _FakeDataset:
                self.operations.append("unbatch")
                return self

            def shuffle(self, buffer_size: int, *, reshuffle_each_iteration: bool) -> _FakeDataset:
                self.operations.append(("shuffle", buffer_size, reshuffle_each_iteration))
                return self

            def batch(self, batch_size: int) -> _FakeDataset:
                self.operations.append(("batch", batch_size))
                return self

            def repeat(self) -> _FakeDataset:
                self.operations.append("repeat")
                return self

            def prefetch(self, buffer_size: object) -> _FakeDataset:
                self.operations.append(("prefetch", buffer_size))
                return self

        class _FakeTfData:
            AUTOTUNE = "autotune"
            Dataset = _FakeDataset

        def _tensor_spec(*, shape: tuple[object, ...], dtype: object) -> tuple[tuple[object, ...], object]:
            return shape, dtype

        class _FakeTensorflow:
            float32 = "float32"
            int32 = "int32"
            data = _FakeTfData
            TensorSpec = staticmethod(_tensor_spec)

        def _fake_import_module(name: str) -> type[_FakeTensorflow]:
            assert name == "tensorflow"
            return _FakeTensorflow

        monkeypatch.setattr(keras_model_module, "import_module", _fake_import_module)

        dataset = MegaritisCnnWeartimeModel(shuffle_buffer_size=32)._make_tf_dataset(
            [(_sensor_data(60), _weartime_list([(0, 60)]))],
            sampling_rate_hz=1.0,
            batch_size=4,
            window_samples=20,
        )

        assert dataset.operations[1:] == [
            "unbatch",
            ("shuffle", 32, True),
            ("batch", 4),
            "repeat",
            ("prefetch", "autotune"),
        ]


class TestWtdMegaritisCNN:
    """Test the CNN detector behavior with deterministic model outputs."""

    def test_detect_requires_model(self) -> None:
        """Require callers to pass the low-level model explicitly."""
        with pytest.raises(RuntimeError, match="Pass a Keras wear-time `model`"):
            WtdMegaritisCNN().detect(_sensor_data(60), sampling_rate_hz=1.0)

    def test_self_optimize_requires_model(self) -> None:
        """Require callers to pass the trainable low-level model explicitly."""
        training_data = [(_sensor_data(60), _weartime_list([(0, 60)]))]

        with pytest.raises(RuntimeError, match="Pass an untrained Keras wear-time `model`"):
            WtdMegaritisCNN().self_optimize(training_data, sampling_rate_hz=1.0, recording_sample_counts=(60,))

    def test_self_optimize_delegates_to_configured_window_model(self) -> None:
        """Delegate training to the low-level Keras model instance."""
        training_data = [(_sensor_data(60), _weartime_list([(0, 60)]))]
        returned_model = MegaritisCnnWeartimeModel()
        model = _SelfOptimizeRecorder(returned_model=returned_model)

        result = WtdMegaritisCNN(model=model).self_optimize(
            training_data,
            sampling_rate_hz=1.0,
            recording_sample_counts=(60,),
        )

        assert model.optimize_training_data is training_data
        assert model.optimize_sampling_rate_hz == 1.0
        assert model.optimize_recording_sample_counts == (60,)
        assert result.model is returned_model

    def test_fixed_window_predictions_regression(self) -> None:
        """Convert deterministic window predictions to expected wear-time intervals."""
        result = WtdMegaritisCNN(
            model=_keras_window_model([1, 1, 0, 0, 1, 1]),
            waking_hours_min=(0, 2),
        ).detect(
            _sensor_data(120),
            sampling_rate_hz=1.0,
        )

        assert_frame_equal(result.weartime_list_, _weartime_list([(0, 40), (80, 120)]))
        assert result.total_weartime_samples_ == 80
        assert result.total_weartime_min_ == pytest.approx(80 / 60)
        assert result.total_weartime_during_waking_min_ == pytest.approx(80 / 60)

    def test_does_not_expose_duplicate_total_weartime_units(self) -> None:
        """Expose only the common base-class wear-time summary metrics."""
        result = WtdMegaritisCNN(
            model=_keras_window_model([1, 1, 0, 0, 1, 1]),
            waking_hours_min=(0, 2),
        ).detect(
            _sensor_data(120),
            sampling_rate_hz=1.0,
        )

        assert not hasattr(result, "total_weartime_minutes_")
        assert not hasattr(result, "total_weartime_hours_")
        assert not hasattr(result, "total_weartime_hours_during_waking_")

    def test_short_ambiguous_interior_wear_bout_is_filtered(self) -> None:
        """Remove short interior wear bouts with insufficient wear-vote confidence."""
        sample_labels = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32)
        vote_counts = np.ones((len(sample_labels), 2), dtype=np.int32)

        result = filter_short_wear_bouts_by_confidence(
            wear_intervals=flags_to_intervals(sample_labels),
            vote_counts=vote_counts,
            data_length=len(sample_labels),
            sampling_rate_hz=1.0,
            min_confidence_short_bouts=0.90,
            short_bout_threshold_minutes=20,
            min_bout_duration_seconds=3,
        )

        assert_array_equal(result, np.empty((0, 2), dtype=np.int64))

    @pytest.mark.parametrize(
        "filename",
        [
            "cnn_lowback_model.keras",
            "cnn_lowback_model_metadata.json",
            "cnn_lstm_lowback_model.keras",
            "cnn_lstm_lowback_metadata.json",
        ],
    )
    def test_model_files_are_packaged(self, filename: str) -> None:
        """Keep the production model artifacts available as package resources."""
        assert keras_model_module.files("mobgap.weartime.production_models").joinpath(filename).is_file()
