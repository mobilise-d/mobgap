"""Tests for the Megaritis XGBoost wear-time detector."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy.signal import welch
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from tpcp.testing import TestAlgorithmMixin

from mobgap.consts import BF_SENSOR_COLS
from mobgap.weartime import WtdMegaritisXGBoost
from mobgap.weartime import _wtd_megaritis_xgboost as xgb_module
from mobgap.weartime.utils.feature_extraction import (
    FEATURE_ORDER_90PCT,
    FULL_FEATURE_ORDER,
    _batched_iqr,
    _batched_spectral_slope,
    _batched_top_peak_freqs,
    _batched_welch,
    extract_features_90pct_batched,
    extract_features_batched,
    extract_full_features_batched,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FixedProbabilityClassifier:
    """Small sklearn-style classifier double returning fixed wear probabilities."""

    batch_sizes: ClassVar[list[int]] = []

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        start = sum(type(self).batch_sizes)
        end = start + len(features)
        type(self).batch_sizes.append(len(features))
        wear_probabilities = self.probabilities[start:end]
        return np.column_stack([1 - wear_probabilities, wear_probabilities])


class _TrainableProbabilityClassifier:
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


class _BrokenTagsUnfittedClassifier:
    """Classifier double matching the XGBoost 1.7/sklearn 1.6 fitted-check failure."""

    def __sklearn_tags__(self) -> None:
        raise AttributeError("'super' object has no attribute '__sklearn_tags__'")

    def get_booster(self) -> None:
        raise NotFittedError("need to call fit or load_model beforehand")

    def fit(self, features: pd.DataFrame, labels: np.ndarray, **_: object) -> _BrokenTagsUnfittedClassifier:
        self.fit_features_ = features.copy()
        self.fit_labels_ = labels.copy()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        wear_probabilities = np.ones(len(features), dtype=np.float32)
        return np.column_stack([1 - wear_probabilities, wear_probabilities])


class _IndexedTrainingData:
    """Lazy training-data double exposing indexed datapoint loading."""

    def __init__(self, records: list[tuple[pd.DataFrame, pd.DataFrame]], *, allow_iter: bool = True) -> None:
        self.records = records
        self.allow_iter = allow_iter
        self.loaded_indices: list[int] = []

    def __iter__(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        if not self.allow_iter:
            raise AssertionError("Parallel indexed training must use `load_recording` instead of `__iter__`.")
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def load_recording(self, datapoint_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.loaded_indices.append(datapoint_index)
        return self.records[datapoint_index]


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
            feature_names=("window_start",),
            window_sec=20.0,
            overlap=0.0,
            waking_hours_min=(0, 2),
            trained_sampling_rate_hz=1.0,
        ).detect(_sensor_data(60), sampling_rate_hz=1.0)


class TestWtdMegaritisXGBoost:
    """Test the XGBoost detector behavior with deterministic classifier outputs."""

    @pytest.mark.parametrize("overlap", [-0.1, 1.0])
    def test_detect_rejects_invalid_overlap(self, overlap: float) -> None:
        """Reject window strides that leave gaps or cannot advance."""
        with pytest.raises(ValueError, match="overlap"):
            WtdMegaritisXGBoost(
                clf=_FixedProbabilityClassifier([0, 0, 0]),
                feature_names=("window_start",),
                window_sec=20.0,
                overlap=overlap,
                trained_sampling_rate_hz=1.0,
            ).detect(_sensor_data(60), sampling_rate_hz=1.0)

    def test_fixed_window_predictions_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Convert deterministic window predictions to expected wear-time intervals."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []
        clf = _FixedProbabilityClassifier([1, 1, 0, 0, 1, 1])

        result = WtdMegaritisXGBoost(
            clf=clf,
            feature_names=("window_start",),
            window_sec=20.0,
            overlap=0.0,
            window_batch_size=2,
            waking_hours_min=(0, 2),
            trained_sampling_rate_hz=1.0,
        ).detect(_sensor_data(120), sampling_rate_hz=1.0)

        assert_frame_equal(result.weartime_list_, _weartime_list([(0, 40), (80, 120)]))
        assert_array_equal(result.window_predictions_, np.array([1, 1, 0, 0, 1, 1], dtype=np.int32))
        assert_array_equal(
            result.window_start_end_,
            np.array([[0, 20], [20, 40], [40, 60], [60, 80], [80, 100], [100, 120]]),
        )
        assert list(result.feature_matrix_.columns) == ["window_start"]
        assert _FixedProbabilityClassifier.batch_sizes == [2, 2, 2]
        assert result.total_weartime_samples_ == 80
        assert result.total_weartime_min_ == pytest.approx(80 / 60)
        assert result.total_weartime_during_waking_min_ == pytest.approx(80 / 60)

    @pytest.mark.parametrize("n_samples", [10, 19])
    def test_no_full_windows_returns_empty_result(self, monkeypatch: pytest.MonkeyPatch, n_samples: int) -> None:
        """Handle recordings shorter than one feature window."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []

        result = WtdMegaritisXGBoost(
            clf=_FixedProbabilityClassifier([]),
            feature_names=("window_start",),
            window_sec=20.0,
            overlap=0.0,
            waking_hours_min=(0, 2),
            trained_sampling_rate_hz=1.0,
        ).detect(_sensor_data(n_samples), sampling_rate_hz=1.0)

        assert_frame_equal(result.weartime_list_, _weartime_list([]))
        assert result.feature_matrix_.empty
        assert result.window_predictions_.size == 0
        assert result.total_weartime_min_ == 0

    def test_self_optimize_handles_xgboost_sklearn_tag_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Train XGBoost-like estimators whose sklearn fitted check fails before fitting."""
        _patch_simple_features(monkeypatch)
        clf = _BrokenTagsUnfittedClassifier()
        training_data = [(_sensor_data(20), _weartime_list([(0, 20)]))]

        result = WtdMegaritisXGBoost(
            clf=clf,
            feature_names=("window_start",),
            window_sec=20.0,
            overlap=0.0,
            trained_sampling_rate_hz=None,
        ).self_optimize(training_data, sampling_rate_hz=1.0, recording_sample_counts=(20,))

        assert result.clf is clf
        assert_array_equal(clf.fit_labels_, np.array([1], dtype=np.int32))

    def test_sampling_rate_is_passed_to_feature_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use the runtime sampling rate for spectral and jerk features."""
        calls: list[dict[str, float]] = []
        _patch_simple_features(monkeypatch, calls)
        _FixedProbabilityClassifier.batch_sizes = []

        WtdMegaritisXGBoost(
            clf=_FixedProbabilityClassifier([1, 0]),
            feature_names=("window_start",),
            window_sec=2.0,
            overlap=0.0,
            waking_hours_min=(0, 1),
            trained_sampling_rate_hz=10.0,
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
            clf=_FixedProbabilityClassifier([1, 0]),
            feature_names=("acc_pa_std",),
            window_sec=20.0,
            overlap=0.0,
            waking_hours_min=(0, 1),
            trained_sampling_rate_hz=1.0,
        ).detect(_sensor_data(40), sampling_rate_hz=1.0)

        assert len(calls) == 1
        assert_array_equal(calls[0], np.array([[0, 20], [20, 40]]))

    def test_self_optimize_parallel_datapoints_preserves_recording_order(self) -> None:
        """Extract indexed training datapoints with joblib and keep fit rows ordered."""
        clf = _TrainableProbabilityClassifier()
        training_data = _IndexedTrainingData(
            [
                (_sensor_data_with_acc_pa_window_means([1.0, 2.0], 20), _weartime_list([(0, 20)])),
                (_sensor_data_with_acc_pa_window_means([3.0, 4.0], 20), _weartime_list([(20, 40)])),
            ],
            allow_iter=False,
        )

        result = WtdMegaritisXGBoost(
            clf=clf,
            feature_names=("acc_pa_mean",),
            window_sec=20.0,
            overlap=0.0,
            window_batch_size=1,
            n_jobs=2,
            trained_sampling_rate_hz=None,
        ).self_optimize(training_data, sampling_rate_hz=1.0, recording_sample_counts=(40, 40))

        assert result.clf is clf
        assert_allclose(clf.fit_features_["acc_pa_mean"].to_numpy(), np.array([1.0, 2.0, 3.0, 4.0]))
        assert_array_equal(clf.fit_labels_, np.array([1, 0, 0, 1], dtype=np.int32))

    def test_self_optimize_fits_classifier_from_window_center_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Train from lazy recording-level data and center labels."""
        _patch_simple_features(monkeypatch)
        clf = _TrainableProbabilityClassifier()
        training_data = [(_sensor_data(60), _weartime_list([(0, 40)]))]

        result = WtdMegaritisXGBoost(
            clf=clf,
            feature_names=("window_start",),
            window_sec=20.0,
            overlap=0.0,
            trained_sampling_rate_hz=None,
        ).self_optimize(training_data, sampling_rate_hz=1.0, recording_sample_counts=(60,), sample_weight="ok")

        assert result.clf is clf
        assert result.trained_sampling_rate_hz == 1.0
        assert result.feature_names == ("window_start",)
        assert_array_equal(clf.fit_features_["window_start"].to_numpy(), np.array([0.0, 20.0, 40.0]))
        assert_array_equal(clf.fit_labels_, np.array([1, 1, 0], dtype=np.int32))
        assert clf.fit_kwargs_ == {"sample_weight": "ok"}

    def test_self_optimize_rejects_mismatching_recording_sample_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep preallocated training arrays aligned with the lazy records."""
        _patch_simple_features(monkeypatch)
        training_data = [(_sensor_data(60), _weartime_list([(0, 40)]))]

        with pytest.raises(ValueError, match="recording_sample_counts"):
            WtdMegaritisXGBoost(
                clf=_TrainableProbabilityClassifier(),
                feature_names=("window_start",),
                window_sec=20.0,
                overlap=0.0,
                trained_sampling_rate_hz=None,
            ).self_optimize(training_data, sampling_rate_hz=1.0, recording_sample_counts=(40,))

    def test_does_not_expose_duplicate_total_weartime_units(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Expose only the common base-class wear-time summary metrics."""
        _patch_simple_features(monkeypatch)
        _FixedProbabilityClassifier.batch_sizes = []

        result = WtdMegaritisXGBoost(
            clf=_FixedProbabilityClassifier([1]),
            feature_names=("window_start",),
            window_sec=20.0,
            overlap=0.0,
            waking_hours_min=(0, 1),
            trained_sampling_rate_hz=1.0,
        ).detect(_sensor_data(20), sampling_rate_hz=1.0)

        assert not hasattr(result, "total_weartime_minutes_")
        assert not hasattr(result, "total_weartime_hours_")
        assert not hasattr(result, "total_weartime_hours_during_waking_")

    @pytest.mark.parametrize(
        "filename",
        [
            "xgboost_90pct_lowback_feature_order.pkl",
            "xgboost_90pct_lowback_metadata.json",
            "xgboost_90pct_lowback_model.pkl",
            "xgboost_fullfeatures_lowback_feature_order.pkl",
            "xgboost_fullfeatures_lowback_metadata.json",
            "xgboost_fullfeatures_lowback_model.pkl",
        ],
    )
    def test_model_files_are_packaged(self, filename: str) -> None:
        """Keep the production model artifacts available as package resources."""
        assert xgb_module.files("mobgap.weartime.production_models").joinpath(filename).is_file()


class TestWtdMegaritisXGBoostFeatureExtraction:
    """Verify classifier probability mapping and extracted feature contracts."""

    @pytest.mark.parametrize("class_label", [0, 1])
    def test_single_class_classifier_maps_wear_probability(self, class_label: int) -> None:
        """Map one-class sklearn probabilities to the wear label."""
        features = pd.DataFrame({"feature": [0.0, 1.0]})
        clf = DummyClassifier(strategy="most_frequent").fit(features, [class_label, class_label])

        probabilities = xgb_module._wear_probabilities(clf, features)

        assert_array_equal(probabilities, np.full(2, class_label, dtype=np.float32))

    """Tests for feature extraction helpers around optional dependencies."""

    def test_pretrained_feature_names_are_available_without_xgboost(self) -> None:
        """Load feature-order resources without importing the optional model package."""
        feature_names = xgb_module._load_pickle_resource("xgboost_90pct_lowback_feature_order.pkl")

        assert feature_names[0] == "acc_pa_std"
        assert len(feature_names) == 79

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
