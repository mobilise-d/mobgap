"""Tests for the Megaritis XGBoost wear-time detector."""

from __future__ import annotations

import json
import pickle
from datetime import time
from threading import get_ident
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest
from joblib import Memory, parallel_config
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from tpcp.misc import warning_error_context
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.weartime import WearTimeModelConfig, WtdMegaritisXGBoost, WtdMegaritisXGBoostTrainer
from mobgap.weartime import _wtd_megaritis_xgboost as xgb_module
from mobgap.weartime import _wtd_megaritis_xgboost_trainer as trainer_module
from mobgap.weartime.utils.feature_extraction import (
    FEATURE_ORDER_90PCT,
    FULL_FEATURE_ORDER,
    _batched_iqr,
    _batched_spectral_slope,
    _batched_time_statistics,
    _batched_top_peak_freqs,
    _batched_welch,
    extract_features_90pct_batched,
    extract_features_batched,
    extract_full_features_batched,
)
from tests.test_weartime.test_wtd_pipeline import DummyDatapoint

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class _FixedProbabilityClassifier(BaseEstimator):
    """Small sklearn-style classifier double returning fixed wear probabilities."""

    batch_sizes: ClassVar[list[int]] = []

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True

    def fit(self, _features: pd.DataFrame, _labels: np.ndarray) -> _FixedProbabilityClassifier:
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        start = sum(type(self).batch_sizes)
        end = start + len(features)
        type(self).batch_sizes.append(len(features))
        wear_probabilities = self.probabilities[start:end]
        return np.column_stack([1 - wear_probabilities, wear_probabilities])


class _TrainableProbabilityClassifier(BaseEstimator):
    """Small sklearn-style classifier double that records fit inputs."""

    def fit(self, features: pd.DataFrame, labels: np.ndarray, **kwargs: object) -> _TrainableProbabilityClassifier:
        self.fit_features_ = features.copy()
        self.fit_labels_ = labels.copy()
        self.fit_kwargs_ = kwargs
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        wear_probabilities = np.ones(len(features), dtype=np.float32)
        return np.column_stack([1 - wear_probabilities, wear_probabilities])


def _dataset(records: list[tuple[pd.DataFrame, pd.DataFrame]], sampling_rate_hz: float = 1.0) -> list[DummyDatapoint]:
    return [
        DummyDatapoint(data=data, reference_weartime=reference, sampling_rate_hz=sampling_rate_hz)
        for data, reference in records
    ]


def _sensor_data(n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(np.ones((n_samples, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)


def _sensor_data_with_acc_pa_window_means(window_means: list[float], window_samples: int) -> pd.DataFrame:
    data = _sensor_data(len(window_means) * window_samples)
    ramp = np.linspace(-0.5, 0.5, window_samples)
    for window_index, window_mean in enumerate(window_means):
        start = window_index * window_samples
        end = start + window_samples
        data.iloc[start:end] = ramp[:, None] + np.arange(len(BF_SENSOR_COLS), dtype=float)
        data.loc[start : end - 1, "acc_pa"] = window_mean + ramp
    return data


def _weartime_list(intervals: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(intervals, columns=["start", "end"]).rename_axis(index="wt_id").astype("int64")


def _reference(intervals: Iterable[tuple[int, int]], uncertain: Iterable[tuple[int, int]] = ()) -> pd.DataFrame:
    reference = pd.concat(
        [_weartime_list(intervals).assign(label="wear"), _weartime_list(uncertain).assign(label="uncertain")],
        ignore_index=True,
    ).rename_axis("wt_id")
    reference["label"] = reference["label"].astype(pd.CategoricalDtype(["wear", "uncertain"]))
    return reference


def _patch_simple_features(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, float]] | None = None) -> None:
    def _extract_features(
        _data: pd.DataFrame,
        window_start_end: np.ndarray,
        *,
        fs: float,
        dt: float,
        feature_names: tuple[str, ...],
        **_: object,
    ) -> pd.DataFrame:
        if calls is not None:
            calls.append({"fs": fs, "dt": dt})
        return pd.DataFrame({feature_name: window_start_end[:, 0].astype(float) for feature_name in feature_names})

    monkeypatch.setattr(xgb_module, "extract_features_batched", _extract_features)


class TestMetaWtdMegaritisXGBoost(TestAlgorithmMixin):
    """Test tpcp algorithm compatibility."""

    __test__ = True

    ALGORITHM_CLASS = WtdMegaritisXGBoost

    @pytest.fixture
    def after_action_instance(self, monkeypatch: pytest.MonkeyPatch) -> WtdMegaritisXGBoost:
        """Create a detector after action for the algorithm mixin."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []
        return self.ALGORITHM_CLASS(
            clf=_FixedProbabilityClassifier([0, 0, 0]),
            model_config=WearTimeModelConfig(1, 20, feature_names=("window_start",)),
            overlap=0.0,
            waking_hours=(time(0), time(0, 2)),
        ).detect(_sensor_data(60), sampling_rate_hz=1.0)


class TestWtdMegaritisXGBoost:
    """Test the XGBoost detector behavior with deterministic classifier outputs."""

    @pytest.mark.parametrize("overlap", [-0.1, 1.0])
    def test_detect_rejects_invalid_overlap(self, overlap: float) -> None:
        """Reject window strides that leave gaps or cannot advance."""
        with pytest.raises(ValueError, match="overlap"):
            WtdMegaritisXGBoost(
                model_config=WearTimeModelConfig(1.0, 20.0, feature_names=("window_start",)),
                clf=_FixedProbabilityClassifier([0, 0, 0]),
                overlap=overlap,
            ).detect(_sensor_data(60), sampling_rate_hz=1.0)

    def test_fixed_window_predictions_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Convert deterministic window predictions to expected wear-time intervals."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []
        clf = _FixedProbabilityClassifier([1, 1, 0, 0, 1, 1])

        result = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(1.0, 20.0, feature_names=("window_start",)),
            clf=clf,
            overlap=0.0,
            window_batch_size=2,
            waking_hours=(time(0), time(0, 2)),
        ).detect(_sensor_data(120), sampling_rate_hz=1.0)

        assert_frame_equal(result.weartime_list_, _weartime_list([(0, 40), (80, 120)]))
        assert_array_equal(result.window_predictions_, np.array([1, 1, 0, 0, 1, 1], dtype=np.int32))
        assert_array_equal(
            result.window_start_end_,
            np.array([[0, 20], [20, 40], [40, 60], [60, 80], [80, 100], [100, 120]]),
        )
        assert _FixedProbabilityClassifier.batch_sizes == [2, 2, 2]
        assert result.total_weartime_samples_ == 80
        assert result.total_weartime_min_ == pytest.approx(80 / 60)
        assert result.total_weartime_during_waking_min_ == pytest.approx(80 / 60)

    def test_probability_at_threshold_counts_as_wear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use the same inclusive probability threshold as the CNN detector."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []
        result = WtdMegaritisXGBoost(
            clf=_FixedProbabilityClassifier([0.5]),
            model_config=WearTimeModelConfig(1, 20, feature_names=("window_start",)),
            overlap=0,
        ).detect(_sensor_data(20), sampling_rate_hz=1)
        assert_array_equal(result.window_predictions_, [1])

    def test_localized_signal_adds_interval_datetimes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Attach local timestamps to detected intervals."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []
        data = _sensor_data(120)
        data.index = pd.date_range("2026-03-29 12:00", periods=120, freq="s", tz="Europe/London")
        result = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(1.0, 20.0, feature_names=("window_start",)),
            clf=_FixedProbabilityClassifier([1, 1, 0, 0, 1, 1]),
            overlap=0.0,
        ).detect(data, sampling_rate_hz=1.0)

        assert result.weartime_list_.local_datetime_start.iloc[0] == data.index[0]
        assert result.weartime_list_.local_datetime_end.iloc[-1] == data.index[-1] + pd.Timedelta(seconds=1)

    @pytest.mark.parametrize("n_samples", [10, 19])
    def test_no_full_windows_returns_empty_result(self, monkeypatch: pytest.MonkeyPatch, n_samples: int) -> None:
        """Handle recordings shorter than one feature window."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []

        result = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(1.0, 20.0, feature_names=("window_start",)),
            clf=_FixedProbabilityClassifier([]),
            overlap=0.0,
            waking_hours=(time(0), time(0, 2)),
        ).detect(_sensor_data(n_samples), sampling_rate_hz=1.0)

        assert_frame_equal(result.weartime_list_, _weartime_list([]))
        assert result.window_predictions_.size == 0
        assert result.total_weartime_min_ == 0

    def test_sampling_rate_is_passed_to_feature_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use the runtime sampling rate for spectral and jerk features."""
        calls: list[dict[str, float]] = []
        _patch_simple_features(monkeypatch, calls)
        _FixedProbabilityClassifier.batch_sizes = []

        WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(10.0, 2.0, feature_names=("window_start",)),
            clf=_FixedProbabilityClassifier([1, 0]),
            overlap=0.0,
            waking_hours=(time(0), time(0, 1)),
        ).detect(_sensor_data(40), sampling_rate_hz=10.0)

        assert calls == [{"fs": 10.0, "dt": 0.1}]

    def test_detector_uses_batched_feature_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use the batched feature path for detector feature extraction."""
        calls: list[np.ndarray] = []

        def _extract_batched(
            _data: pd.DataFrame,
            window_start_end: np.ndarray,
            *,
            feature_names: tuple[str, ...],
            version: str,
            **_kwargs: object,
        ) -> pd.DataFrame:
            calls.append(window_start_end.copy())
            assert version == "lightweight"
            return pd.DataFrame({feature_name: np.ones(len(window_start_end)) for feature_name in feature_names})

        monkeypatch.setattr(xgb_module, "extract_features_batched", _extract_batched)
        _FixedProbabilityClassifier.batch_sizes = []

        WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(1.0, 20.0, feature_names=("acc_pa_std",)),
            clf=_FixedProbabilityClassifier([1, 0]),
            overlap=0.0,
            waking_hours=(time(0), time(0, 1)),
        ).detect(_sensor_data(40), sampling_rate_hz=1.0)

        assert len(calls) == 1
        assert_array_equal(calls[0], np.array([[0, 20], [20, 40]]))

    def test_self_optimize_parallel_datapoints_preserves_recording_order(self) -> None:
        """Extract indexed training datapoints with joblib and keep fit rows ordered."""
        clf = _TrainableProbabilityClassifier()
        training_data = _dataset(
            [
                (_sensor_data_with_acc_pa_window_means([1.0, 2.0], 20), _reference([(0, 20)])),
                (_sensor_data_with_acc_pa_window_means([3.0, 4.0], 20), _reference([(20, 40)])),
            ],
        )

        result = WtdMegaritisXGBoostTrainer(
            algo=WtdMegaritisXGBoost(overlap=0.0, window_batch_size=1, n_jobs_features=2),
            classifier=clf,
            feature_names=("acc_pa_mean",),
            window_sec=20.0,
            n_jobs=2,
            training_overlap=0.0,
            convert_to_body_frame=False,
        ).self_optimize(training_data)

        assert result.algo.clf is not clf
        clf = result.algo.clf
        assert_allclose(clf.fit_features_["acc_pa_mean"].to_numpy(), np.array([1.0, 2.0, 3.0, 4.0]))
        assert_array_equal(clf.fit_labels_, np.array([1, 0, 0, 1], dtype=np.int32))

    @pytest.mark.parametrize(
        ("uncertain", "expected_starts", "expected_labels"),
        [
            (None, [0, 10, 20, 30, 40], [1, 1, 1, 0, 0]),
            ([(35, 60)], [0, 10], [1, 1]),
            ([(20, 30)], [0, 30, 40], [1, 0, 0]),
            ([(10, 40)], [40], [0]),
        ],
    )
    def test_self_optimize_fits_classifier_from_window_center_labels(
        self,
        monkeypatch: pytest.MonkeyPatch,
        uncertain: list[tuple[int, int]] | None,
        expected_starts: list[int],
        expected_labels: list[int],
    ) -> None:
        """Exclude intersecting windows without shifting centers; skip entirely unknown recordings."""
        calls = []
        _patch_simple_features(monkeypatch, calls)
        clf = _TrainableProbabilityClassifier()
        recording = (_sensor_data(60), _reference([(0, 40)]))
        if uncertain is not None:
            recording = (
                recording[0],
                _reference(recording[1][["start", "end"]].itertuples(index=False, name=None), uncertain),
            )
        training_data = [recording, (_sensor_data(60), _reference([], [(0, 60)]))]

        result = WtdMegaritisXGBoostTrainer(
            algo=WtdMegaritisXGBoost(overlap=0.5, window_batch_size=2),
            classifier=clf,
            feature_names=("window_start",),
            window_sec=20.0,
            training_overlap=0.5,
            convert_to_body_frame=False,
        ).self_optimize(_dataset(training_data), sample_weight="ok")

        assert result.algo.clf is not clf
        clf = result.algo.clf
        assert result.algo.model_config.sampling_rate_hz == 1.0
        assert result.feature_names == ("window_start",)
        assert_array_equal(clf.fit_features_["window_start"].to_numpy(), expected_starts)
        assert_array_equal(clf.fit_labels_, expected_labels)
        assert clf.fit_kwargs_ == {"sample_weight": "ok"}
        assert len(calls) == 3  # Full-recording features remain cached independently of uncertainty.

    def test_self_optimize_reuses_full_recording_features(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Repeated fits reuse the recording's features while retaining the training labels."""
        calls: list[dict[str, float]] = []
        _patch_simple_features(monkeypatch, calls)
        data = _sensor_data(60)
        changed_data = data.copy()
        changed_data.loc[0, "acc_pa"] = 2.0
        memory = Memory(tmp_path, verbose=0)

        for recording, intervals, expected_labels, expected_extractions in [
            (data, [(0, 40)], [1, 1, 0], 2),
            (data, [(20, 60)], [0, 1, 1], 2),
            (changed_data, [(0, 40)], [1, 1, 0], 4),
        ]:
            clf = _TrainableProbabilityClassifier()
            model = WtdMegaritisXGBoostTrainer(
                algo=WtdMegaritisXGBoost(overlap=0.0, window_batch_size=2, memory=memory),
                classifier=clf,
                feature_names=("window_start",),
                window_sec=20.0,
                training_overlap=0.0,
                convert_to_body_frame=False,
            ).self_optimize(_dataset([(recording, _reference(intervals))]))
            assert_array_equal(model.algo.clf.fit_labels_, np.array(expected_labels, dtype=np.int32))
            model.algo.detect(recording, sampling_rate_hz=1.0)
            assert len(calls) == expected_extractions

    def test_does_not_expose_duplicate_total_weartime_units(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Expose only the common base-class wear-time summary metrics."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []

        result = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(1.0, 20.0, feature_names=("window_start",)),
            clf=_FixedProbabilityClassifier([1]),
            overlap=0.0,
            waking_hours=(time(0), time(0, 1)),
        ).detect(_sensor_data(20), sampling_rate_hz=1.0)

        assert not hasattr(result, "total_weartime_minutes_")
        assert not hasattr(result, "total_weartime_hours_")
        assert not hasattr(result, "total_weartime_hours_during_waking_")

    def test_sustain_preset_loads_selected_hyperparameters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load the newly trained classifier, feature order and JSON parameters together."""
        xgboost = pytest.importorskip("xgboost")
        features = pd.DataFrame({"acc_pa_std": [0.0, 1.0, 2.0, 3.0]})
        classifier = xgboost.XGBClassifier(n_estimators=1, max_depth=2, n_jobs=1).fit(features, [0, 0, 1, 1])
        with (tmp_path / "xgboost_sustain_weartime_model.pkl").open("wb") as file:
            pickle.dump(classifier, file)
        (tmp_path / "xgboost_sustain_weartime_model_config.json").write_text(
            json.dumps(WearTimeModelConfig(50, feature_names=("acc_pa_std",)).to_dict())
        )
        (tmp_path / "xgboost_sustain_weartime_best_params.json").write_text(
            json.dumps({"classifier__max_depth": 2, "classifier__n_estimators": 1})
        )
        monkeypatch.setattr(xgb_module, "files", lambda _: tmp_path)
        xgb_module._load_pickle_resource.cache_clear()
        config = WtdMegaritisXGBoost.PredefinedParameters.sustain_weartime
        check_is_fitted(config["clf"])
        assert config["clf"].max_depth == 2
        assert config["clf"].n_estimators == 1
        assert config["model_config"].feature_names == ("acc_pa_std",)
        assert config["model_config"].sampling_rate_hz == 50
        monkeypatch.setattr(trainer_module, "files", lambda _: tmp_path)
        template = WtdMegaritisXGBoostTrainer.PredefinedParameters.sustain_weartime
        assert template["classifier"].max_depth == 2
        assert template["classifier"].n_estimators == 1
        assert_array_equal(config["clf"].predict_proba(features), classifier.predict_proba(features))
        xgb_module._load_pickle_resource.cache_clear()

    def test_fractional_sampling_rate_uses_actual_window_center(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Round window duration once and label at the midpoint of that actual window."""
        _patch_simple_features(monkeypatch)
        trainer = WtdMegaritisXGBoostTrainer(
            classifier=_TrainableProbabilityClassifier(),
            window_sec=5,
            feature_names=("window_start",),
            training_overlap=0,
            convert_to_body_frame=False,
        ).self_optimize(_dataset([(_sensor_data(500), _reference([(250, 500)]))], 99.9))
        assert_array_equal(trainer.algo.clf.fit_labels_, [1])

    def test_trainer_keeps_inference_overlap_and_requires_native_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Training stride does not replace inference stride or silently change sampling rates."""
        _patch_simple_features(monkeypatch)
        template = _TrainableProbabilityClassifier()
        trainer = WtdMegaritisXGBoostTrainer(
            WtdMegaritisXGBoost(overlap=0.5),
            classifier=template,
            window_sec=20,
            feature_names=("window_start",),
            training_overlap=0,
            convert_to_body_frame=False,
        ).self_optimize(_dataset([(_sensor_data(60), _reference([(0, 60)]))]))
        assert not hasattr(template, "classes_")
        assert len(trainer.algo.clf.fit_features_) == 3
        assert trainer.algo.overlap == 0.5
        trainer.run(_dataset([(_sensor_data(60), _reference([(0, 60)]))])[0])
        assert len(trainer.algo_.window_predictions_) == 5
        with pytest.raises(ValueError, match=r"requires 1\.0 Hz"):
            trainer.algo.detect(_sensor_data(60), sampling_rate_hz=2)


class TestWtdMegaritisXGBoostFeatureExtraction:
    """Verify extracted feature contracts."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_fused_statistics_match_reference(self, dtype: type) -> None:
        """Preserve corrected moments, sample variance and constant/unknown windows."""
        values = np.random.default_rng(42).normal(size=(8, 500)).astype(dtype)
        values[0] = 0
        values[1] = 7
        values[2] += 1e6
        values[3, 0] = np.nan
        values[4] = 0.1
        reference = values.astype(np.float64)
        with np.errstate(invalid="ignore"):
            expected = (
                reference.mean(axis=1),
                reference.std(axis=1, ddof=1),
                np.sqrt((reference * reference).mean(axis=1)),
                reference.min(axis=1),
                reference.max(axis=1),
                skew(reference, axis=1, bias=False),
                kurtosis(reference, axis=1, bias=False, fisher=False),
                (reference * reference).mean(axis=1),
            )
        for actual, target in zip(_batched_time_statistics(values), expected):
            assert_allclose(actual, target, rtol=1e-8, atol=1e-8)

    def test_near_constant_moments_respect_scipy_zero_convention(self) -> None:
        """Preserve both near-zero conventions used by supported SciPy releases."""
        values = np.tile([1.0, 1.0 + 2**-50], (1, 250))
        current = _batched_time_statistics(values, zero_tolerance=np.finfo(np.float64).eps)
        legacy = _batched_time_statistics(values, zero_tolerance=np.finfo(np.float64).resolution)
        assert np.isfinite(current.kurtosis).all()
        assert np.isnan(legacy.kurtosis).all()

    @pytest.mark.parametrize("version", ["lightweight", "full"])
    def test_symmetric_gyro_mean_preserves_dominant_axis_ratio(self, version: str) -> None:
        """Do not amplify summation residuals through a near-zero dominant-axis denominator."""
        amplitude = 1024 + 2**-39
        values = np.zeros((500, 6), dtype=np.float64)
        values[:, 3] = np.tile([amplitude, -amplitude], 250)
        features = extract_features_batched(
            values,
            np.array([[0, 500]]),
            feature_names=("gyr_is_mean", "time_gyr_norm_to_dominant_ratio"),
            version=version,
        )
        assert features.iloc[0, 0] == 0
        assert_allclose(features.iloc[0, 1], amplitude / 1e-12, rtol=1e-12)

    @pytest.mark.parametrize("version", ["lightweight", "full"])
    def test_small_axis_motion_survives_float32_storage(self, version: str) -> None:
        """Calculate norms wide enough to retain representable low-intensity axis movements."""
        values = np.zeros((500, 6), dtype=np.float32)
        values[:, 3] = 1
        values[1::2, 4] = 2**-13
        features = extract_features_batched(
            values,
            np.array([[0, 500]]),
            feature_names=("gyr_norm_perm_entropy", "burst_count_norm"),
            version=version,
        )
        assert_allclose(features.iloc[0], [1 / np.log2(6), 249], rtol=1e-12)

    @pytest.mark.parametrize("version", ["lightweight", "full"])
    def test_near_constant_float32_moments_match_float64_reference(self, version: str) -> None:
        """Retain finite corrected moments for small motion around a large gyroscope baseline."""
        values = np.zeros((500, 6), dtype=np.float32)
        values[:, 3] = 100
        values[0, 3] += 2**-17
        features = extract_features_batched(
            values, np.array([[0, 500]]), feature_names=("gyr_is_kurtosis",), version=version
        )
        expected = kurtosis(values[:, 3].astype(np.float64), fisher=False, bias=False)
        assert_allclose(features.iloc[0, 0], expected, rtol=1e-8)

    def test_nullable_sensor_frame_matches_nan_array(self) -> None:
        """Keep nullable pandas sensors compatible with missing-value feature extraction."""
        data = pd.DataFrame(np.ones((625, 6), dtype=np.float32), columns=BF_SENSOR_COLS).astype("Float32")
        data.iloc[0, 0] = pd.NA
        algo = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(100.0, 5.0, feature_names=FEATURE_ORDER_90PCT),
        )
        expected = algo.extract_features(data.astype(np.float64), 100)
        assert_frame_equal(algo.extract_features(data, 100), expected)

    @pytest.mark.parametrize("step", [125, 500])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_recording_features_match_independent_windows(self, step: int, dtype: type) -> None:
        """Reuse overlapping and non-overlapping intermediates without shifting batches or ordinal patterns."""
        values = np.random.default_rng(42).normal(size=(1750, 6)).astype(dtype)
        values[300:400] = 0  # Repeated values exercise the ordinal tie-breaking rules.
        data = pd.DataFrame(values, columns=BF_SENSOR_COLS)
        starts = np.arange(0, len(data) - 499, step)
        windows = np.column_stack([starts, starts + 500])
        expected = extract_features_batched(data, windows).to_numpy(dtype=np.float32)
        algo = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(100.0, 5.0, feature_names=FEATURE_ORDER_90PCT),
            window_batch_size=2,
            overlap=1 - step / 500,
            n_jobs_features=1,
        )
        assert_array_equal(algo.extract_features(data, 100).to_numpy(), expected)
        parallel = algo.clone().set_params(n_jobs_features=2)
        assert_array_equal(parallel.extract_features(data, 100).to_numpy(), expected)

    @pytest.mark.parametrize(("recording_jobs", "feature_jobs"), [(1, 1), (1, 2)])
    def test_feature_threads_are_independent_of_recording_processes(
        self, recording_jobs: int, feature_jobs: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training and inference use feature threads regardless of the recording process count."""
        data = pd.DataFrame(np.zeros((625, 6), dtype=np.float32), columns=BF_SENSOR_COLS)
        caller_thread = get_ident()
        worker_threads = []
        extract = xgb_module.extract_features_batched

        def record_thread(*args: object, **kwargs: object) -> pd.DataFrame:
            worker_threads.append(get_ident())
            return extract(*args, **kwargs)

        monkeypatch.setattr(xgb_module, "extract_features_batched", record_thread)
        algo = WtdMegaritisXGBoostTrainer(
            algo=WtdMegaritisXGBoost(window_batch_size=1, n_jobs_features=feature_jobs),
            classifier=_TrainableProbabilityClassifier(),
            feature_names=("acc_pa_mean",),
            n_jobs=recording_jobs,
            training_overlap=0.75,
            convert_to_body_frame=False,
        )
        algo.self_optimize(_dataset([(data, _reference([(0, 625)]))], 100))
        assert worker_threads
        assert all((thread == caller_thread) == (feature_jobs == 1) for thread in worker_threads)
        worker_threads.clear()
        algo.algo.extract_features(data, 100)
        assert worker_threads
        assert all((thread == caller_thread) == (feature_jobs == 1) for thread in worker_threads)

    def test_worker_count_shares_recording_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A threaded cache hit reuses features previously extracted serially."""
        data = pd.DataFrame(np.zeros((625, 6), dtype=np.float32), columns=BF_SENSOR_COLS)
        algo = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(100.0, 5.0, feature_names=FEATURE_ORDER_90PCT),
            memory=Memory(tmp_path, verbose=0),
            n_jobs_features=1,
        )
        expected = algo.extract_features(data, 100)

        def unexpected_extraction(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Worker count must not invalidate the recording cache")

        monkeypatch.setattr(xgb_module, "_prepare_lightweight_recording", unexpected_extraction)
        assert_frame_equal(algo.clone().set_params(n_jobs_features=2).extract_features(data, 100), expected)

    def test_threaded_extraction_preserves_warning_context_under_process_preference(self) -> None:
        """Keep caller warning records and allow an outer joblib process preference."""
        data = pd.DataFrame(np.ones((625, 6), dtype=np.float32), columns=BF_SENSOR_COLS)
        algo = WtdMegaritisXGBoost(
            model_config=WearTimeModelConfig(100.0, 2, feature_names=FEATURE_ORDER_90PCT),
            window_batch_size=1,
            n_jobs_features=2,
        )
        with (
            parallel_config(prefer="processes"),
            warning_error_context("recording", {"id": "sample"}, record_only=True) as context,
        ):
            features = algo.extract_features(data, 100)
        assert len(features) > 1
        assert context.records
        assert all("recording: id='sample'" in record.context for record in context.records)
        assert any("Input window duration" in str(record.message) for record in context.records)

    def test_batched_lightweight_features_match_full_subset(self) -> None:
        """Keep the lightweight extractor equivalent to the full batched extractor for shared features."""
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(size=(875, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)
        window_start_end = np.array([[0, 500], [125, 625], [250, 750], [375, 875]])

        lightweight = extract_features_90pct_batched(data, window_start_end, fs=100.0, dt=0.01)
        full_subset = extract_full_features_batched(
            data,
            window_start_end,
            fs=100.0,
            dt=0.01,
            feature_names=FEATURE_ORDER_90PCT,
        )

        assert list(lightweight.columns) == FEATURE_ORDER_90PCT
        assert_allclose(lightweight.to_numpy(), full_subset.to_numpy(), rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("dtype", [np.int16, np.float16])
    def test_batched_features_match_for_numeric_array_and_dataframe(self, dtype: type) -> None:
        """Array input uses the same floating-point feature calculations as a DataFrame."""
        rng = np.random.default_rng(42)
        values = rng.integers(900, 1100, size=(625, len(BF_SENSOR_COLS))).astype(dtype)
        data = pd.DataFrame(values, columns=BF_SENSOR_COLS)
        windows = np.array([[0, 500], [125, 625]])
        feature_names = ("acc_norm_rms", "gyr_pa_rms")

        from_dataframe = extract_features_batched(data, windows, feature_names=feature_names)
        from_array = extract_features_batched(values, windows, feature_names=feature_names)

        assert_allclose(from_array.to_numpy(), from_dataframe.to_numpy(), rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(("column", "baseline"), [(3, 100.0), (0, 9.81)])
    def test_full_features_preserve_small_float32_rolling_variance(self, column: int, baseline: float) -> None:
        """Retain small movement variability around a large baseline with float32 sensor storage."""
        values = np.zeros((500, len(BF_SENSOR_COLS)), dtype=np.float32)
        values[:, column] = baseline
        values[::3, column] += 2**-10
        gyr_windows = np.lib.stride_tricks.sliding_window_view(values[:, 3].astype(np.float64), 10)
        acc_windows = np.lib.stride_tricks.sliding_window_view(values[:, 0].astype(np.float64), 10)
        expected = [np.var(gyr_windows, axis=1, ddof=1).mean(), np.var(acc_windows, axis=1, ddof=1).std(ddof=1)]
        features = extract_full_features_batched(
            values,
            np.array([[0, 500]]),
            feature_names=("rolling_var_mean_norm", "rolling_var_std_norm"),
        )
        assert_allclose(features.iloc[0], expected, rtol=1e-8, atol=1e-14)

    def test_batched_full_features_cover_full_feature_order(self) -> None:
        """Extract every full-model feature without falling back to scalar feature calculators."""
        rng = np.random.default_rng(1234)
        data = pd.DataFrame(rng.normal(size=(625, len(BF_SENSOR_COLS))), columns=BF_SENSOR_COLS)
        window_start_end = np.array([[0, 500], [125, 625]])

        full_features = extract_features_batched(
            data,
            window_start_end,
            fs=100.0,
            dt=0.01,
            feature_names=FULL_FEATURE_ORDER,
            version="full",
        )

        assert list(full_features.columns) == FULL_FEATURE_ORDER
        assert full_features.shape == (2, len(FULL_FEATURE_ORDER))
        assert np.isfinite(full_features.to_numpy()).all()

    def test_batched_peak_frequencies_match_find_peaks_plateau_behavior(self) -> None:
        """Match scipy find_peaks for flat-topped PSD peaks."""
        frequencies = np.arange(5, dtype=np.float64)
        power = np.array(
            [
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 2.0, 0.0],
            ]
        )

        peak_freqs = _batched_top_peak_freqs(frequencies, power, 2)

        assert_allclose(peak_freqs, np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))

    def test_batched_iqr_preserves_percentile_nan_behavior(self) -> None:
        """Keep NaN handling equivalent to the scalar percentile feature."""
        values = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, np.nan, 3.0, 4.0]])
        percentile = np.percentile(values, [75, 25], axis=1)

        assert_allclose(_batched_iqr(values), percentile[0] - percentile[1])

    def test_float32_iqr_keeps_fractional_order_statistics(self) -> None:
        """Do not erase a small IQR by rounding both interpolated quartiles to the same float32 value."""
        values = np.full((1, 500), 100, dtype=np.float32)
        values[:, :125] = np.nextafter(np.float32(100), np.float32(-np.inf))
        values[:, -125:] = np.nextafter(np.float32(100), np.float32(np.inf))
        percentiles = np.percentile(values.astype(np.float64), [25, 75], axis=1)
        assert_allclose(_batched_iqr(values), percentiles[1] - percentiles[0], rtol=1e-12)

    def test_batched_spectral_slope_matches_short_spectrum_guard(self) -> None:
        """Return zero when there are too few valid positive frequency bins."""
        frequencies = np.array([0.0, 1.0, 2.0])
        power = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])

        assert_array_equal(_batched_spectral_slope(frequencies, power), np.array([0.0, 0.0]))

    def test_batched_welch_matches_scipy_welch(self) -> None:
        """Keep the specialized batched Welch helper equivalent to scipy.signal.welch."""
        rng = np.random.default_rng(123)
        values = rng.normal(size=(4, 50))

        frequencies, power = _batched_welch(values, fs=100.0)
        expected_frequencies, expected_power = welch(values, fs=100.0, nperseg=values.shape[1], axis=1)

        assert_array_equal(frequencies, expected_frequencies)
        assert_allclose(power, expected_power, rtol=1e-14, atol=1e-14)

    def test_float32_constant_welch_has_no_artificial_power(self) -> None:
        """Detrend stationary float32 signals without inventing low-frequency motion."""
        values = np.full((2, 500), 1.5 + 2**-17, dtype=np.float32)
        _, power = _batched_welch(values, fs=100)
        assert_array_equal(power, np.zeros_like(power))
