"""Utility functions for IMU feature extraction from windowed data."""

import types
import warnings
from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyentrp import entropy as ent
from scipy.signal import coherence, find_peaks, welch
from scipy.stats import kurtosis, skew


# ---------- Helper functions ----------
def rms(x: Union[NDArray[np.float64], Sequence[float]]) -> float:
    """Calculate root mean square."""
    return np.sqrt(np.mean(x**2)) if len(x) else 0.0


def zero_crossing_rate(x: Union[NDArray[np.float64], Sequence[float]]) -> float:
    """Calculate zero-crossing rate."""
    return np.mean(np.diff(np.signbit(x)) != 0) if len(x) > 1 else 0.0


def consecutive_true_lengths(mask: NDArray[np.bool_]) -> NDArray[np.int_]:
    """Find lengths of consecutive True values in boolean mask."""
    if len(mask) == 0:
        return np.array([])
    diff = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts


def _reorder_features(
    features: dict[str, float],
    expected_order: list[str],
) -> dict[str, float]:
    """Reorder features dict to match expected order from training."""
    return {k: features[k] for k in expected_order if k in features}


def psd_features(
    x: np.ndarray,
    prefix: str,
    fs: float,
    lf_band: tuple[float, float],
    ent: types.ModuleType,
) -> dict[str, float]:
    """
    Compute spectral features from a 1D signal using power spectral density.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    prefix : str
        Feature name prefix.
    fs : float
        Sampling frequency.
    lf_band : tuple[float, float]
        Low-frequency band definition.
    ent : module
        Entropy computation module.

    Returns
    -------
    dict[str, float]
        Extracted spectral features.
    """
    features: dict[str, float] = {}

    f, pxx = welch(x, fs=fs, nperseg=len(x))
    total_power = np.sum(pxx)
    features[f"{prefix}_psd_total"] = total_power

    lf_mask = (f >= lf_band[0]) & (f <= lf_band[1])
    hf_mask = f > lf_band[1]

    lf_power = np.sum(pxx[lf_mask])
    hf_power = np.sum(pxx[hf_mask])

    features[f"{prefix}_lf_power"] = lf_power
    features[f"{prefix}_lf_hf_ratio"] = lf_power / hf_power if hf_power > 0 else 0.0

    peaks, _ = find_peaks(pxx)
    peak_freqs = f[peaks][np.argsort(pxx[peaks])[::-1]] if len(peaks) > 0 else np.array([])

    for k, name in enumerate(["dom_freq", "second_peak_freq", "third_peak_freq"]):
        features[f"{prefix}_{name}"] = peak_freqs[k] if len(peak_freqs) > k else 0.0

    if total_power > 0:
        psd_norm = pxx / total_power
        features[f"{prefix}_spectral_centroid"] = np.sum(f * psd_norm)
        features[f"{prefix}_spectral_entropy"] = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    else:
        features[f"{prefix}_spectral_centroid"] = 0.0
        features[f"{prefix}_spectral_entropy"] = 0.0

    features[f"{prefix}_spectral_skewness"] = skew(pxx, bias=False)
    features[f"{prefix}_spectral_kurtosis"] = kurtosis(pxx, fisher=False, bias=False)

    valid = (f > 0) & (pxx > 0)
    features[f"{prefix}_spectral_slope"] = (
        np.polyfit(np.log(f[valid]), np.log(pxx[valid]), 1)[0] if np.sum(valid) > 2 else 0.0
    )

    features[f"{prefix}_perm_entropy"] = ent.permutation_entropy(x, order=3, normalize=True)

    return features


def _add_coherence_features(
    features: dict[str, float],
    data: np.ndarray,
    axes: Sequence[str],
    fs: float,
    n: int,
) -> None:
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            _, cxy = coherence(data[:, i], data[:, j], fs=fs, nperseg=n)
            features[f"{axes[i]}_{axes[j]}_coherence_mean"] = np.mean(cxy)


def _add_axis_relation_features(
    features: dict[str, float],
    data: np.ndarray,
    norm: np.ndarray,
    axes: Sequence[str],
) -> None:
    axis_energy = np.mean(data**2, axis=0)
    features["axis_dominance_ratio"] = np.max(axis_energy) / np.sum(axis_energy) if np.sum(axis_energy) > 0 else 0.0

    dominant_idx = np.argmax(axis_energy)
    features["freq_gyr_norm_to_dominant_ratio"] = np.mean(norm) / (np.mean(data[:, dominant_idx]) + 1e-12)

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            features[f"{axes[i]}_{axes[j]}_psd_ratio"] = np.sum(np.abs(np.fft.rfft(data[:, i]))) / (
                np.sum(np.abs(np.fft.rfft(data[:, j]))) + 1e-12
            )


def _add_gyr_psd_features(
    features: dict[str, float],
    data: np.ndarray,
    norm: np.ndarray,
    axes: Sequence[str],
    fs: float,
    lf_band: tuple[float, float],
) -> None:
    for i, ax in enumerate(axes):
        features.update(psd_features(data[:, i], ax, fs, lf_band, ent))

    features.update(psd_features(norm, "gyr_norm", fs, lf_band, ent))


# Full feature order
FULL_FEATURE_ORDER = [
    "gyr_is_mean",
    "gyr_is_std",
    "gyr_is_rms",
    "gyr_is_skew",
    "gyr_is_kurtosis",
    "gyr_is_range",
    "gyr_is_iqr",
    "gyr_is_zcr",
    "gyr_is_jerk_mean",
    "gyr_is_jerk_rms",
    "gyr_is_jerk_peak",
    "gyr_ml_mean",
    "gyr_ml_std",
    "gyr_ml_rms",
    "gyr_ml_skew",
    "gyr_ml_kurtosis",
    "gyr_ml_range",
    "gyr_ml_iqr",
    "gyr_ml_zcr",
    "gyr_ml_jerk_mean",
    "gyr_ml_jerk_rms",
    "gyr_ml_jerk_peak",
    "gyr_pa_mean",
    "gyr_pa_std",
    "gyr_pa_rms",
    "gyr_pa_skew",
    "gyr_pa_kurtosis",
    "gyr_pa_range",
    "gyr_pa_iqr",
    "gyr_pa_zcr",
    "gyr_pa_jerk_mean",
    "gyr_pa_jerk_rms",
    "gyr_pa_jerk_peak",
    "gyr_norm_mean",
    "gyr_norm_std",
    "gyr_norm_rms",
    "gyr_norm_range",
    "gyr_norm_iqr",
    "gyr_norm_skew",
    "gyr_norm_kurtosis",
    "gyr_jerk_rms_norm",
    "gyr_jerk_peak_norm",
    "pct_above_noise_norm",
    "duty_cycle_norm",
    "median_near_zero_duration_norm",
    "burst_count_norm",
    "rolling_var_mean_norm",
    "moving_range_mean_norm",
    "corr_is_ml",
    "corr_is_pa",
    "corr_ml_pa",
    "axis_dominance_ratio",
    "time_gyr_norm_to_dominant_ratio",
    "gyr_is_gyr_ml_energy_ratio",
    "gyr_is_gyr_pa_energy_ratio",
    "gyr_ml_gyr_pa_energy_ratio",
    "acc_is_mean",
    "acc_is_std",
    "acc_is_rms",
    "acc_is_median",
    "acc_is_min",
    "acc_is_max",
    "acc_is_range",
    "acc_is_iqr",
    "acc_is_skew",
    "acc_is_kurtosis",
    "acc_is_zcr",
    "acc_is_jerk_mean",
    "acc_is_jerk_rms",
    "acc_is_jerk_peak",
    "acc_is_jerk_std",
    "acc_ml_mean",
    "acc_ml_std",
    "acc_ml_rms",
    "acc_ml_median",
    "acc_ml_min",
    "acc_ml_max",
    "acc_ml_range",
    "acc_ml_iqr",
    "acc_ml_skew",
    "acc_ml_kurtosis",
    "acc_ml_zcr",
    "acc_ml_jerk_mean",
    "acc_ml_jerk_rms",
    "acc_ml_jerk_peak",
    "acc_ml_jerk_std",
    "acc_pa_mean",
    "acc_pa_std",
    "acc_pa_rms",
    "acc_pa_median",
    "acc_pa_min",
    "acc_pa_max",
    "acc_pa_range",
    "acc_pa_iqr",
    "acc_pa_skew",
    "acc_pa_kurtosis",
    "acc_pa_zcr",
    "acc_pa_jerk_mean",
    "acc_pa_jerk_rms",
    "acc_pa_jerk_peak",
    "acc_pa_jerk_std",
    "acc_norm_mean",
    "acc_norm_std",
    "acc_norm_rms",
    "acc_norm_median",
    "acc_norm_min",
    "acc_norm_max",
    "acc_norm_range",
    "acc_norm_iqr",
    "acc_norm_skew",
    "acc_norm_kurtosis",
    "acc_jerk_rms_norm",
    "acc_jerk_peak_norm",
    "acc_jerk_std_norm",
    "max_near_zero_duration_norm",
    "rolling_var_std_norm",
    "time_acc_norm_to_dominant_ratio",
    "acc_is_acc_ml_energy_ratio",
    "acc_is_acc_pa_energy_ratio",
    "acc_ml_acc_pa_energy_ratio",
    "gyr_is_psd_total",
    "gyr_is_lf_power",
    "gyr_is_lf_hf_ratio",
    "gyr_is_dom_freq",
    "gyr_is_second_peak_freq",
    "gyr_is_third_peak_freq",
    "gyr_is_spectral_centroid",
    "gyr_is_spectral_entropy",
    "gyr_is_spectral_skewness",
    "gyr_is_spectral_kurtosis",
    "gyr_is_spectral_slope",
    "gyr_is_perm_entropy",
    "gyr_ml_psd_total",
    "gyr_ml_lf_power",
    "gyr_ml_lf_hf_ratio",
    "gyr_ml_dom_freq",
    "gyr_ml_second_peak_freq",
    "gyr_ml_third_peak_freq",
    "gyr_ml_spectral_centroid",
    "gyr_ml_spectral_entropy",
    "gyr_ml_spectral_skewness",
    "gyr_ml_spectral_kurtosis",
    "gyr_ml_spectral_slope",
    "gyr_ml_perm_entropy",
    "gyr_pa_psd_total",
    "gyr_pa_lf_power",
    "gyr_pa_lf_hf_ratio",
    "gyr_pa_dom_freq",
    "gyr_pa_second_peak_freq",
    "gyr_pa_third_peak_freq",
    "gyr_pa_spectral_centroid",
    "gyr_pa_spectral_entropy",
    "gyr_pa_spectral_skewness",
    "gyr_pa_spectral_kurtosis",
    "gyr_pa_spectral_slope",
    "gyr_pa_perm_entropy",
    "gyr_norm_psd_total",
    "gyr_norm_lf_power",
    "gyr_norm_lf_hf_ratio",
    "gyr_norm_dom_freq",
    "gyr_norm_second_peak_freq",
    "gyr_norm_third_peak_freq",
    "gyr_norm_spectral_centroid",
    "gyr_norm_spectral_entropy",
    "gyr_norm_spectral_skewness",
    "gyr_norm_spectral_kurtosis",
    "gyr_norm_spectral_slope",
    "gyr_norm_perm_entropy",
    "gyr_is_gyr_ml_coherence_mean",
    "gyr_is_gyr_pa_coherence_mean",
    "gyr_ml_gyr_pa_coherence_mean",
    "freq_gyr_norm_to_dominant_ratio",
    "gyr_is_gyr_ml_psd_ratio",
    "gyr_is_gyr_pa_psd_ratio",
    "gyr_ml_gyr_pa_psd_ratio",
    "acc_is_psd_total",
    "acc_is_lf_power",
    "acc_is_lf_hf_ratio",
    "acc_is_dom_freq",
    "acc_is_second_peak_freq",
    "acc_is_third_peak_freq",
    "acc_is_spectral_centroid",
    "acc_is_spectral_entropy",
    "acc_is_spectral_skewness",
    "acc_is_spectral_kurtosis",
    "acc_is_spectral_slope",
    "acc_is_perm_entropy",
    "acc_ml_psd_total",
    "acc_ml_lf_power",
    "acc_ml_lf_hf_ratio",
    "acc_ml_dom_freq",
    "acc_ml_second_peak_freq",
    "acc_ml_third_peak_freq",
    "acc_ml_spectral_centroid",
    "acc_ml_spectral_entropy",
    "acc_ml_spectral_skewness",
    "acc_ml_spectral_kurtosis",
    "acc_ml_spectral_slope",
    "acc_ml_perm_entropy",
    "acc_pa_psd_total",
    "acc_pa_lf_power",
    "acc_pa_lf_hf_ratio",
    "acc_pa_dom_freq",
    "acc_pa_second_peak_freq",
    "acc_pa_third_peak_freq",
    "acc_pa_spectral_centroid",
    "acc_pa_spectral_entropy",
    "acc_pa_spectral_skewness",
    "acc_pa_spectral_kurtosis",
    "acc_pa_spectral_slope",
    "acc_pa_perm_entropy",
    "acc_norm_psd_total",
    "acc_norm_lf_power",
    "acc_norm_lf_hf_ratio",
    "acc_norm_dom_freq",
    "acc_norm_second_peak_freq",
    "acc_norm_third_peak_freq",
    "acc_norm_spectral_centroid",
    "acc_norm_spectral_entropy",
    "acc_norm_spectral_skewness",
    "acc_norm_spectral_kurtosis",
    "acc_norm_spectral_slope",
    "acc_norm_perm_entropy",
    "acc_is_acc_ml_coherence_mean",
    "acc_is_acc_pa_coherence_mean",
    "acc_ml_acc_pa_coherence_mean",
    "acc_is_acc_ml_psd_ratio",
    "acc_is_acc_pa_psd_ratio",
    "acc_ml_acc_pa_psd_ratio",
    "freq_acc_norm_to_dominant_ratio",
]

# 95% feature order
FEATURE_ORDER_95PCT = [
    "acc_pa_std",
    "gyr_ml_spectral_centroid",
    "gyr_is_spectral_centroid",
    "time_acc_norm_to_dominant_ratio",
    "gyr_ml_lf_power",
    "gyr_ml_spectral_slope",
    "acc_pa_spectral_centroid",
    "gyr_is_spectral_slope",
    "gyr_pa_mean",
    "acc_is_jerk_mean",
    "gyr_is_std",
    "gyr_is_gyr_pa_energy_ratio",
    "gyr_ml_mean",
    "acc_is_mean",
    "axis_dominance_ratio",
    "gyr_pa_rms",
    "gyr_ml_iqr",
    "acc_is_min",
    "acc_pa_rms",
    "acc_is_acc_pa_psd_ratio",
    "acc_is_median",
    "acc_jerk_std_norm",
    "gyr_is_mean",
    "gyr_ml_lf_hf_ratio",
    "acc_is_max",
    "acc_norm_spectral_centroid",
    "gyr_ml_gyr_pa_energy_ratio",
    "acc_pa_median",
    "acc_pa_second_peak_freq",
    "acc_is_spectral_slope",
    "acc_is_acc_ml_psd_ratio",
    "gyr_is_perm_entropy",
    "gyr_is_gyr_pa_psd_ratio",
    "moving_range_mean_norm",
    "acc_norm_dom_freq",
    "gyr_pa_jerk_mean",
    "acc_is_perm_entropy",
    "acc_ml_mean",
    "gyr_is_jerk_mean",
    "acc_jerk_rms_norm",
    "acc_pa_dom_freq",
    "acc_norm_median",
    "gyr_is_kurtosis",
    "acc_ml_median",
    "gyr_pa_iqr",
    "acc_ml_perm_entropy",
    "acc_pa_mean",
    "acc_jerk_peak_norm",
    "acc_ml_acc_pa_energy_ratio",
    "gyr_norm_spectral_slope",
    "gyr_is_gyr_ml_energy_ratio",
    "acc_pa_max",
    "gyr_is_zcr",
    "acc_pa_perm_entropy",
    "acc_is_acc_ml_energy_ratio",
    "gyr_is_jerk_rms",
    "acc_norm_second_peak_freq",
    "acc_is_jerk_std",
    "gyr_norm_skew",
    "acc_ml_spectral_slope",
    "acc_pa_min",
    "acc_norm_mean",
    "acc_pa_psd_total",
    "acc_pa_third_peak_freq",
    "burst_count_norm",
    "acc_is_zcr",
    "acc_norm_rms",
    "gyr_pa_zcr",
    "acc_ml_rms",
    "acc_pa_spectral_slope",
    "acc_norm_spectral_slope",
    "gyr_ml_jerk_rms",
    "gyr_ml_gyr_pa_psd_ratio",
    "gyr_pa_spectral_entropy",
    "time_gyr_norm_to_dominant_ratio",
    "acc_ml_acc_pa_psd_ratio",
    "gyr_ml_dom_freq",
    "gyr_is_rms",
    "gyr_norm_perm_entropy",
    "gyr_norm_kurtosis",
    "acc_norm_third_peak_freq",
    "corr_ml_pa",
    "acc_is_jerk_peak",
    "acc_ml_max",
    "gyr_ml_std",
    "acc_pa_jerk_mean",
    "gyr_ml_kurtosis",
    "acc_ml_min",
    "acc_norm_perm_entropy",
    "acc_ml_spectral_centroid",
    "acc_norm_spectral_skewness",
    "acc_is_acc_pa_energy_ratio",
    "acc_norm_iqr",
    "acc_is_rms",
    "gyr_ml_rms",
    "gyr_ml_perm_entropy",
    "acc_is_jerk_rms",
    "acc_norm_min",
    "gyr_is_iqr",
]

# 90% feature order
FEATURE_ORDER_90PCT = [
    "acc_pa_std",
    "gyr_ml_spectral_centroid",
    "gyr_is_spectral_centroid",
    "time_acc_norm_to_dominant_ratio",
    "gyr_ml_lf_power",
    "gyr_ml_spectral_slope",
    "acc_pa_spectral_centroid",
    "gyr_is_spectral_slope",
    "gyr_pa_mean",
    "acc_is_jerk_mean",
    "gyr_is_std",
    "gyr_is_gyr_pa_energy_ratio",
    "gyr_ml_mean",
    "acc_is_mean",
    "axis_dominance_ratio",
    "gyr_pa_rms",
    "gyr_ml_iqr",
    "acc_is_min",
    "acc_pa_rms",
    "acc_is_acc_pa_psd_ratio",
    "acc_is_median",
    "acc_jerk_std_norm",
    "gyr_is_mean",
    "gyr_ml_lf_hf_ratio",
    "acc_is_max",
    "acc_norm_spectral_centroid",
    "gyr_ml_gyr_pa_energy_ratio",
    "acc_pa_median",
    "acc_pa_second_peak_freq",
    "acc_is_spectral_slope",
    "acc_is_acc_ml_psd_ratio",
    "gyr_is_perm_entropy",
    "gyr_is_gyr_pa_psd_ratio",
    "moving_range_mean_norm",
    "acc_norm_dom_freq",
    "gyr_pa_jerk_mean",
    "acc_is_perm_entropy",
    "acc_ml_mean",
    "gyr_is_jerk_mean",
    "acc_jerk_rms_norm",
    "acc_pa_dom_freq",
    "acc_norm_median",
    "gyr_is_kurtosis",
    "acc_ml_median",
    "gyr_pa_iqr",
    "acc_ml_perm_entropy",
    "acc_pa_mean",
    "acc_jerk_peak_norm",
    "acc_ml_acc_pa_energy_ratio",
    "gyr_norm_spectral_slope",
    "gyr_is_gyr_ml_energy_ratio",
    "acc_pa_max",
    "gyr_is_zcr",
    "acc_pa_perm_entropy",
    "acc_is_acc_ml_energy_ratio",
    "gyr_is_jerk_rms",
    "acc_norm_second_peak_freq",
    "acc_is_jerk_std",
    "gyr_norm_skew",
    "acc_ml_spectral_slope",
    "acc_pa_min",
    "acc_norm_mean",
    "acc_pa_psd_total",
    "acc_pa_third_peak_freq",
    "burst_count_norm",
    "acc_is_zcr",
    "acc_norm_rms",
    "gyr_pa_zcr",
    "acc_ml_rms",
    "acc_pa_spectral_slope",
    "acc_norm_spectral_slope",
    "gyr_ml_jerk_rms",
    "gyr_ml_gyr_pa_psd_ratio",
    "gyr_pa_spectral_entropy",
    "time_gyr_norm_to_dominant_ratio",
    "acc_ml_acc_pa_psd_ratio",
    "gyr_ml_dom_freq",
    "gyr_is_rms",
    "gyr_norm_perm_entropy",
]


# Feature extraction pipeline intentionally explicit for traceability/readability.
def extract_full_features(
    df: pd.DataFrame,
    acc_axes: tuple[str, str, str] = ("acc_is", "acc_ml", "acc_pa"),
    gyr_axes: tuple[str, str, str] = ("gyr_is", "gyr_ml", "gyr_pa"),
    fs: float = 100.0,
    dt: float = 0.01,
    lf_band: tuple[float, float] = (0.0, 0.5),
    noise_floor: float = 0.05,
    near_zero_thr: float = 0.02,
    rolling_win: int = 10,
) -> dict[str, float]:
    """
    Extract complete set of time-domain and frequency-domain features from 6-axis IMU data.

    Computes 230 features combining accelerometer and gyroscope signals across
    time and frequency domains. Features include statistical moments, spectral
    characteristics, jerk metrics, activity patterns, and cross-axis correlations.

    Args:
        df: DataFrame containing IMU data with accelerometer and gyroscope columns
        acc_axes: Tuple of accelerometer column names (IS, ML, PA axes)
        gyr_axes: Tuple of gyroscope column names (IS, ML, PA axes)
        fs: Sampling frequency in Hz (default: 100.0)
        dt: Time step in seconds (default: 0.01)
        lf_band: Low-frequency band range in Hz (default: 0.0-0.5)
        noise_floor: Threshold for activity detection (default: 0.05)
        near_zero_thr: Threshold for stillness detection (default: 0.02)
        rolling_win: Window size for rolling statistics (default: 10)

    Returns
    -------
        Dictionary mapping feature names to scalar values (230 features total)

    Notes
    -----
        Model expects 5-second windows at 100 Hz (500 samples). A warning is
        issued if input duration differs from expected window size.
    """
    n_samples = len(df)
    expected_samples = int(5.0 * fs)

    if n_samples != expected_samples:
        warnings.warn(
            f"Input window duration is {n_samples / fs:.2f}s. "
            f"Model was trained on 5.0s windows. Performance may be affected.",
            UserWarning,
            stacklevel=2,
        )

    features = {}
    acc_time = _extract_acc_time_domain(df, acc_axes, dt, noise_floor, near_zero_thr, rolling_win)
    features.update(acc_time)
    acc_freq = _extract_acc_frequency_domain(df, acc_axes, fs, lf_band)
    features.update(acc_freq)
    gyr_time = _extract_gyr_time_domain(df, gyr_axes, dt, noise_floor, near_zero_thr, rolling_win)
    features.update(gyr_time)
    gyr_freq = _extract_gyr_frequency_domain(df, gyr_axes, fs, lf_band)
    features.update(gyr_freq)

    # Returing re order
    return _reorder_features(features, FULL_FEATURE_ORDER)


# Feature extraction pipeline intentionally explicit for traceability/readability.
def extract_features_90pct(  # noqa: PLR0912, PLR0915
    df: pd.DataFrame,
    acc_axes: tuple[str, str, str] = ("acc_is", "acc_ml", "acc_pa"),
    gyr_axes: tuple[str, str, str] = ("gyr_is", "gyr_ml", "gyr_pa"),
    fs: float = 100.0,
    dt: float = 0.01,
    lf_band: tuple[float, float] = (0.0, 0.5),
    rolling_win: int = 10,
) -> dict[str, float]:
    """
    Extract reduced feature set explaining 90% of SHAP importance (79 features).

    Computes a computationally efficient subset of features selected via SHAP
    analysis to explain 90% of model predictions while reducing extraction time.

    Args:
        df: DataFrame containing IMU data with accelerometer and gyroscope columns
        acc_axes: Tuple of accelerometer column names (IS, ML, PA axes)
        gyr_axes: Tuple of gyroscope column names (IS, ML, PA axes)
        fs: Sampling frequency in Hz (default: 100.0)
        dt: Time step in seconds (default: 0.01)
        lf_band: Low-frequency band range in Hz (default: 0.0-0.5)
        rolling_win: Window size for rolling statistics (default: 10)

    Returns
    -------
        Dictionary mapping feature names to scalar values (79 features total)

    Notes
    -----
        Features selected based on SHAP analysis to maximize predictive power
        while minimizing computational cost. Model expects 5-second windows at 100 Hz.
    """
    n_samples = len(df)
    expected_samples = int(5.0 * fs)
    if n_samples != expected_samples:
        warnings.warn(
            f"Input window duration is {n_samples / fs:.2f}s. "
            f"Model was trained on 5.0s windows. Performance may be affected.",
            UserWarning,
            stacklevel=2,
        )

    features = {}
    acc_data = df[list(acc_axes)].to_numpy(dtype=float)
    gyr_data = df[list(gyr_axes)].to_numpy(dtype=float)
    acc_norm = np.linalg.norm(acc_data, axis=1)
    gyr_norm = np.linalg.norm(gyr_data, axis=1)

    # Accelerometer IS axis
    features["acc_is_mean"] = np.mean(acc_data[:, 0])
    features["acc_is_median"] = np.median(acc_data[:, 0])
    features["acc_is_min"] = np.min(acc_data[:, 0])
    features["acc_is_max"] = np.max(acc_data[:, 0])
    features["acc_is_zcr"] = zero_crossing_rate(acc_data[:, 0])
    jerk_is = np.diff(acc_data[:, 0]) / dt
    features["acc_is_jerk_mean"] = np.mean(np.abs(jerk_is))
    features["acc_is_jerk_std"] = np.std(jerk_is, ddof=1)
    f_is, pxx_is = welch(acc_data[:, 0], fs=fs, nperseg=len(acc_data[:, 0]))
    valid_is = (f_is > 0) & (pxx_is > 0)
    features["acc_is_spectral_slope"] = (
        np.polyfit(np.log(f_is[valid_is]), np.log(pxx_is[valid_is]), 1)[0] if np.sum(valid_is) > 2 else 0.0
    )
    features["acc_is_perm_entropy"] = ent.permutation_entropy(acc_data[:, 0], order=3, normalize=True)

    # Accelerometer ML axis
    features["acc_ml_mean"] = np.mean(acc_data[:, 1])
    features["acc_ml_median"] = np.median(acc_data[:, 1])
    features["acc_ml_rms"] = rms(acc_data[:, 1])
    f_ml, pxx_ml = welch(acc_data[:, 1], fs=fs, nperseg=len(acc_data[:, 1]))
    valid_ml = (f_ml > 0) & (pxx_ml > 0)
    features["acc_ml_spectral_slope"] = (
        np.polyfit(np.log(f_ml[valid_ml]), np.log(pxx_ml[valid_ml]), 1)[0] if np.sum(valid_ml) > 2 else 0.0
    )
    features["acc_ml_perm_entropy"] = ent.permutation_entropy(acc_data[:, 1], order=3, normalize=True)

    # Accelerometer PA axis
    features["acc_pa_mean"] = np.mean(acc_data[:, 2])
    features["acc_pa_median"] = np.median(acc_data[:, 2])
    features["acc_pa_min"] = np.min(acc_data[:, 2])
    features["acc_pa_max"] = np.max(acc_data[:, 2])
    features["acc_pa_std"] = np.std(acc_data[:, 2], ddof=1)
    features["acc_pa_rms"] = rms(acc_data[:, 2])
    f_pa, pxx_pa = welch(acc_data[:, 2], fs=fs, nperseg=len(acc_data[:, 2]))
    total_pa = np.sum(pxx_pa)
    features["acc_pa_psd_total"] = total_pa
    if total_pa > 0:
        psd_pa_norm = pxx_pa / total_pa
        features["acc_pa_spectral_centroid"] = np.sum(f_pa * psd_pa_norm)
    else:
        features["acc_pa_spectral_centroid"] = 0.0
    peaks_pa, _ = find_peaks(pxx_pa)
    if len(peaks_pa) > 0:
        peak_freqs_pa = f_pa[peaks_pa][np.argsort(pxx_pa[peaks_pa])[::-1]]
        features["acc_pa_dom_freq"] = peak_freqs_pa[0] if len(peak_freqs_pa) > 0 else 0.0
        features["acc_pa_second_peak_freq"] = peak_freqs_pa[1] if len(peak_freqs_pa) > 1 else 0.0
        features["acc_pa_third_peak_freq"] = peak_freqs_pa[2] if len(peak_freqs_pa) > 2 else 0.0
    else:
        features["acc_pa_dom_freq"] = 0.0
        features["acc_pa_second_peak_freq"] = 0.0
        features["acc_pa_third_peak_freq"] = 0.0
    valid_pa = (f_pa > 0) & (pxx_pa > 0)
    features["acc_pa_spectral_slope"] = (
        np.polyfit(np.log(f_pa[valid_pa]), np.log(pxx_pa[valid_pa]), 1)[0] if np.sum(valid_pa) > 2 else 0.0
    )
    features["acc_pa_perm_entropy"] = ent.permutation_entropy(acc_data[:, 2], order=3, normalize=True)

    # Accelerometer norm
    features["acc_norm_mean"] = np.mean(acc_norm)
    features["acc_norm_median"] = np.median(acc_norm)
    features["acc_norm_rms"] = rms(acc_norm)
    f_norm, pxx_norm = welch(acc_norm, fs=fs, nperseg=len(acc_norm))
    total_norm = np.sum(pxx_norm)
    if total_norm > 0:
        psd_norm_acc = pxx_norm / total_norm
        features["acc_norm_spectral_centroid"] = np.sum(f_norm * psd_norm_acc)
    else:
        features["acc_norm_spectral_centroid"] = 0.0
    peaks_norm, _ = find_peaks(pxx_norm)
    if len(peaks_norm) > 0:
        peak_freqs_norm = f_norm[peaks_norm][np.argsort(pxx_norm[peaks_norm])[::-1]]
        features["acc_norm_dom_freq"] = peak_freqs_norm[0] if len(peak_freqs_norm) > 0 else 0.0
        features["acc_norm_second_peak_freq"] = peak_freqs_norm[1] if len(peak_freqs_norm) > 1 else 0.0
    else:
        features["acc_norm_dom_freq"] = 0.0
        features["acc_norm_second_peak_freq"] = 0.0
    valid_norm = (f_norm > 0) & (pxx_norm > 0)
    features["acc_norm_spectral_slope"] = (
        np.polyfit(np.log(f_norm[valid_norm]), np.log(pxx_norm[valid_norm]), 1)[0] if np.sum(valid_norm) > 2 else 0.0
    )

    # Accelerometer jerk norm
    jerk_acc_norm = np.diff(acc_norm) / dt
    features["acc_jerk_rms_norm"] = rms(jerk_acc_norm)
    features["acc_jerk_peak_norm"] = np.max(np.abs(jerk_acc_norm))
    features["acc_jerk_std_norm"] = np.std(jerk_acc_norm, ddof=1)

    # Gyroscope activity patterns
    features["burst_count_norm"] = np.sum((gyr_norm[1:-1] > gyr_norm[:-2]) & (gyr_norm[1:-1] > gyr_norm[2:]))
    s_gyr = pd.Series(gyr_norm)
    features["moving_range_mean_norm"] = s_gyr.diff().abs().rolling(rolling_win, min_periods=rolling_win).mean().mean()

    # Accelerometer cross-axis features
    dominant_idx_acc = np.argmax(np.mean(acc_data**2, axis=0))
    features["time_acc_norm_to_dominant_ratio"] = np.mean(acc_norm) / (np.mean(acc_data[:, dominant_idx_acc]) + 1e-12)
    features["acc_is_acc_ml_energy_ratio"] = np.sum(np.abs(acc_data[:, 0])) / (np.sum(np.abs(acc_data[:, 1])) + 1e-12)
    features["acc_ml_acc_pa_energy_ratio"] = np.sum(np.abs(acc_data[:, 1])) / (np.sum(np.abs(acc_data[:, 2])) + 1e-12)
    features["acc_is_acc_ml_psd_ratio"] = np.sum(np.abs(np.fft.rfft(acc_data[:, 0]))) / (
        np.sum(np.abs(np.fft.rfft(acc_data[:, 1]))) + 1e-12
    )
    features["acc_is_acc_pa_psd_ratio"] = np.sum(np.abs(np.fft.rfft(acc_data[:, 0]))) / (
        np.sum(np.abs(np.fft.rfft(acc_data[:, 2]))) + 1e-12
    )
    features["acc_ml_acc_pa_psd_ratio"] = np.sum(np.abs(np.fft.rfft(acc_data[:, 1]))) / (
        np.sum(np.abs(np.fft.rfft(acc_data[:, 2]))) + 1e-12
    )

    # Gyroscope IS axis
    features["gyr_is_mean"] = np.mean(gyr_data[:, 0])
    features["gyr_is_std"] = np.std(gyr_data[:, 0], ddof=1)
    features["gyr_is_rms"] = rms(gyr_data[:, 0])
    features["gyr_is_kurtosis"] = kurtosis(gyr_data[:, 0], fisher=False, bias=False)
    features["gyr_is_zcr"] = zero_crossing_rate(gyr_data[:, 0])
    jerk_gyr_is = np.diff(gyr_data[:, 0]) / dt
    features["gyr_is_jerk_mean"] = np.mean(np.abs(jerk_gyr_is))
    features["gyr_is_jerk_rms"] = rms(jerk_gyr_is)
    f_gyr_is, pxx_gyr_is = welch(gyr_data[:, 0], fs=fs, nperseg=len(gyr_data[:, 0]))
    total_gyr_is = np.sum(pxx_gyr_is)
    if total_gyr_is > 0:
        psd_gyr_is_norm = pxx_gyr_is / total_gyr_is
        features["gyr_is_spectral_centroid"] = np.sum(f_gyr_is * psd_gyr_is_norm)
    else:
        features["gyr_is_spectral_centroid"] = 0.0
    valid_gyr_is = (f_gyr_is > 0) & (pxx_gyr_is > 0)
    features["gyr_is_spectral_slope"] = (
        np.polyfit(np.log(f_gyr_is[valid_gyr_is]), np.log(pxx_gyr_is[valid_gyr_is]), 1)[0]
        if np.sum(valid_gyr_is) > 2
        else 0.0
    )
    features["gyr_is_perm_entropy"] = ent.permutation_entropy(gyr_data[:, 0], order=3, normalize=True)

    # Gyroscope ML axis
    features["gyr_ml_mean"] = np.mean(gyr_data[:, 1])
    features["gyr_ml_iqr"] = np.subtract(*np.percentile(gyr_data[:, 1], [75, 25]))
    jerk_gyr_ml = np.diff(gyr_data[:, 1]) / dt
    features["gyr_ml_jerk_rms"] = rms(jerk_gyr_ml)
    f_gyr_ml, pxx_gyr_ml = welch(gyr_data[:, 1], fs=fs, nperseg=len(gyr_data[:, 1]))
    lf_mask = (f_gyr_ml >= lf_band[0]) & (f_gyr_ml <= lf_band[1])
    hf_mask = f_gyr_ml > lf_band[1]
    lf_power = np.sum(pxx_gyr_ml[lf_mask])
    hf_power = np.sum(pxx_gyr_ml[hf_mask])
    features["gyr_ml_lf_power"] = lf_power
    features["gyr_ml_lf_hf_ratio"] = lf_power / hf_power if hf_power > 0 else 0.0
    total_gyr_ml = np.sum(pxx_gyr_ml)
    if total_gyr_ml > 0:
        psd_gyr_ml_norm = pxx_gyr_ml / total_gyr_ml
        features["gyr_ml_spectral_centroid"] = np.sum(f_gyr_ml * psd_gyr_ml_norm)
    else:
        features["gyr_ml_spectral_centroid"] = 0.0
    valid_gyr_ml = (f_gyr_ml > 0) & (pxx_gyr_ml > 0)
    features["gyr_ml_spectral_slope"] = (
        np.polyfit(np.log(f_gyr_ml[valid_gyr_ml]), np.log(pxx_gyr_ml[valid_gyr_ml]), 1)[0]
        if np.sum(valid_gyr_ml) > 2
        else 0.0
    )
    peaks_gyr_ml, _ = find_peaks(pxx_gyr_ml)
    if len(peaks_gyr_ml) > 0:
        peak_freqs_gyr_ml = f_gyr_ml[peaks_gyr_ml][np.argsort(pxx_gyr_ml[peaks_gyr_ml])[::-1]]
        features["gyr_ml_dom_freq"] = peak_freqs_gyr_ml[0]
    else:
        features["gyr_ml_dom_freq"] = 0.0

    # Gyroscope PA axis
    features["gyr_pa_mean"] = np.mean(gyr_data[:, 2])
    features["gyr_pa_rms"] = rms(gyr_data[:, 2])
    features["gyr_pa_iqr"] = np.subtract(*np.percentile(gyr_data[:, 2], [75, 25]))
    features["gyr_pa_zcr"] = zero_crossing_rate(gyr_data[:, 2])
    jerk_gyr_pa = np.diff(gyr_data[:, 2]) / dt
    features["gyr_pa_jerk_mean"] = np.mean(np.abs(jerk_gyr_pa))
    _, pxx_gyr_pa = welch(gyr_data[:, 2], fs=fs, nperseg=len(gyr_data[:, 2]))
    total_gyr_pa = np.sum(pxx_gyr_pa)
    if total_gyr_pa > 0:
        psd_gyr_pa_norm = pxx_gyr_pa / total_gyr_pa
        features["gyr_pa_spectral_entropy"] = -np.sum(psd_gyr_pa_norm * np.log(psd_gyr_pa_norm + 1e-12))
    else:
        features["gyr_pa_spectral_entropy"] = 0.0

    # Gyroscope norm
    features["gyr_norm_skew"] = skew(gyr_norm, bias=False)
    f_gyr_norm, pxx_gyr_norm = welch(gyr_norm, fs=fs, nperseg=len(gyr_norm))
    valid_gyr_norm = (f_gyr_norm > 0) & (pxx_gyr_norm > 0)
    features["gyr_norm_spectral_slope"] = (
        np.polyfit(np.log(f_gyr_norm[valid_gyr_norm]), np.log(pxx_gyr_norm[valid_gyr_norm]), 1)[0]
        if np.sum(valid_gyr_norm) > 2
        else 0.0
    )
    features["gyr_norm_perm_entropy"] = ent.permutation_entropy(gyr_norm, order=3, normalize=True)

    # Cross-axis coupling
    axis_energy_gyr = np.mean(gyr_data**2, axis=0)
    features["axis_dominance_ratio"] = (
        np.max(axis_energy_gyr) / np.sum(axis_energy_gyr) if np.sum(axis_energy_gyr) > 0 else 0.0
    )
    dominant_idx_gyr = np.argmax(axis_energy_gyr)
    features["time_gyr_norm_to_dominant_ratio"] = np.mean(gyr_norm) / (np.mean(gyr_data[:, dominant_idx_gyr]) + 1e-12)
    features["gyr_is_gyr_ml_energy_ratio"] = np.sum(np.abs(gyr_data[:, 0])) / (np.sum(np.abs(gyr_data[:, 1])) + 1e-12)
    features["gyr_is_gyr_pa_energy_ratio"] = np.sum(np.abs(gyr_data[:, 0])) / (np.sum(np.abs(gyr_data[:, 2])) + 1e-12)
    features["gyr_ml_gyr_pa_energy_ratio"] = np.sum(np.abs(gyr_data[:, 1])) / (np.sum(np.abs(gyr_data[:, 2])) + 1e-12)
    features["gyr_is_gyr_pa_psd_ratio"] = np.sum(np.abs(np.fft.rfft(gyr_data[:, 0]))) / (
        np.sum(np.abs(np.fft.rfft(gyr_data[:, 2]))) + 1e-12
    )
    features["gyr_ml_gyr_pa_psd_ratio"] = np.sum(np.abs(np.fft.rfft(gyr_data[:, 1]))) / (
        np.sum(np.abs(np.fft.rfft(gyr_data[:, 2]))) + 1e-12
    )

    # Return re ordered
    return _reorder_features(features, FEATURE_ORDER_90PCT)


def _extract_acc_time_domain(  # noqa: PLR0915
    df,  # noqa: ANN001
    axes,  # noqa: ANN001
    dt,  # noqa: ANN001
    noise_floor,  # noqa: ANN001
    near_zero_thr,  # noqa: ANN001
    rolling_win,  # noqa: ANN001
) -> dict[str, float]:
    """Extract time-domain features from accelerometer data."""
    features = {}
    if df.empty or any(ax not in df.columns for ax in axes):
        return {f"{ax}_missing": np.nan for ax in axes}

    data = df[list(axes)].to_numpy(dtype=float)
    n = data.shape[0]
    if n < 3:
        return {f"{ax}_too_short": np.nan for ax in axes}

    norm = np.linalg.norm(data, axis=1)

    for i, ax in enumerate(axes):
        x = data[:, i]
        features[f"{ax}_mean"] = np.mean(x)
        features[f"{ax}_std"] = np.std(x, ddof=1)
        features[f"{ax}_rms"] = rms(x)
        features[f"{ax}_median"] = np.median(x)
        features[f"{ax}_min"] = np.min(x)
        features[f"{ax}_max"] = np.max(x)
        features[f"{ax}_range"] = np.ptp(x)
        features[f"{ax}_iqr"] = np.subtract(*np.percentile(x, [75, 25]))
        features[f"{ax}_skew"] = skew(x, bias=False)
        features[f"{ax}_kurtosis"] = kurtosis(x, fisher=False, bias=False)
        features[f"{ax}_zcr"] = zero_crossing_rate(x)

        jerk = np.diff(x) / dt
        features[f"{ax}_jerk_mean"] = np.mean(np.abs(jerk))
        features[f"{ax}_jerk_rms"] = rms(jerk)
        features[f"{ax}_jerk_peak"] = np.max(np.abs(jerk))
        features[f"{ax}_jerk_std"] = np.std(jerk, ddof=1)

    features["acc_norm_mean"] = np.mean(norm)
    features["acc_norm_std"] = np.std(norm, ddof=1)
    features["acc_norm_rms"] = rms(norm)
    features["acc_norm_median"] = np.median(norm)
    features["acc_norm_min"] = np.min(norm)
    features["acc_norm_max"] = np.max(norm)
    features["acc_norm_range"] = np.ptp(norm)
    features["acc_norm_iqr"] = np.subtract(*np.percentile(norm, [75, 25]))
    features["acc_norm_skew"] = skew(norm, bias=False)
    features["acc_norm_kurtosis"] = kurtosis(norm, fisher=False, bias=False)

    jerk_norm = np.diff(norm) / dt
    features["acc_jerk_rms_norm"] = rms(jerk_norm)
    features["acc_jerk_peak_norm"] = np.max(np.abs(jerk_norm))
    features["acc_jerk_std_norm"] = np.std(jerk_norm, ddof=1)

    above_noise = norm > noise_floor
    features["pct_above_noise_norm"] = np.mean(above_noise)
    run_lengths = consecutive_true_lengths(above_noise)
    features["duty_cycle_norm"] = np.sum(run_lengths) / n if len(run_lengths) else 0.0

    near_zero = norm < near_zero_thr
    nz_lengths = consecutive_true_lengths(near_zero)
    features["median_near_zero_duration_norm"] = np.median(nz_lengths) * dt if len(nz_lengths) else 0.0
    features["max_near_zero_duration_norm"] = np.max(nz_lengths) * dt if len(nz_lengths) else 0.0
    features["burst_count_norm"] = np.sum((norm[1:-1] > norm[:-2]) & (norm[1:-1] > norm[2:]))

    s = pd.Series(norm)
    features["rolling_var_mean_norm"] = s.rolling(rolling_win, min_periods=rolling_win).var().mean()
    features["rolling_var_std_norm"] = s.rolling(rolling_win, min_periods=rolling_win).var().std()
    features["moving_range_mean_norm"] = s.diff().abs().rolling(rolling_win, min_periods=rolling_win).mean().mean()

    features["corr_is_ml"] = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    features["corr_is_pa"] = np.corrcoef(data[:, 0], data[:, 2])[0, 1]
    features["corr_ml_pa"] = np.corrcoef(data[:, 1], data[:, 2])[0, 1]

    axis_energy = np.mean(data**2, axis=0)
    features["axis_dominance_ratio"] = np.max(axis_energy) / np.sum(axis_energy) if np.sum(axis_energy) > 0 else 0.0
    dominant_idx = np.argmax(axis_energy)
    features["time_acc_norm_to_dominant_ratio"] = np.mean(norm) / (np.mean(data[:, dominant_idx]) + 1e-12)

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            features[f"{axes[i]}_{axes[j]}_energy_ratio"] = np.sum(np.abs(data[:, i])) / (
                np.sum(np.abs(data[:, j])) + 1e-12
            )

    return features


def _extract_acc_frequency_domain(
    df: pd.DataFrame,
    axes: tuple[str, str, str],
    fs: float,
    lf_band: tuple[float, float],
) -> dict[str, float]:
    """Extract frequency-domain features from accelerometer data."""
    features = {}
    if df.empty or any(ax not in df.columns for ax in axes):
        return {f"{ax}_missing": np.nan for ax in axes}

    data = df[list(axes)].to_numpy(dtype=float)
    n = data.shape[0]
    if n < 8:
        return {f"{ax}_too_short": np.nan for ax in axes}

    norm = np.linalg.norm(data, axis=1)

    for i, ax in enumerate(axes):
        features.update(psd_features(data[:, i], ax, fs, lf_band, ent))

    features.update(psd_features(norm, "acc_norm", fs, lf_band, ent))

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            _, cxy = coherence(data[:, i], data[:, j], fs=fs, nperseg=n)
            features[f"{axes[i]}_{axes[j]}_coherence_mean"] = np.mean(cxy)

    axis_energy = np.mean(data**2, axis=0)
    features["axis_dominance_ratio"] = np.max(axis_energy) / np.sum(axis_energy) if np.sum(axis_energy) > 0 else 0.0
    dominant_idx = np.argmax(axis_energy)
    features["freq_acc_norm_to_dominant_ratio"] = np.mean(norm) / (np.mean(data[:, dominant_idx]) + 1e-12)

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            features[f"{axes[i]}_{axes[j]}_psd_ratio"] = np.sum(np.abs(np.fft.rfft(data[:, i]))) / (
                np.sum(np.abs(np.fft.rfft(data[:, j]))) + 1e-12
            )

    return features


def _extract_gyr_time_domain(
    df: pd.DataFrame,
    axes: Sequence[str],
    dt: float,
    noise_floor: float,
    near_zero_thr: float,
    rolling_win: int,
) -> dict[str, float]:
    """Extract time-domain features from gyroscope data."""
    features = {}
    if df.empty or any(ax not in df.columns for ax in axes):
        return {f"{ax}_missing": np.nan for ax in axes}

    data = df[list(axes)].to_numpy(dtype=float)
    n = data.shape[0]
    if n < 3:
        return {f"{ax}_too_short": np.nan for ax in axes}

    norm = np.linalg.norm(data, axis=1)

    for i, ax in enumerate(axes):
        x = data[:, i]
        features[f"{ax}_mean"] = np.mean(x)
        features[f"{ax}_std"] = np.std(x, ddof=1)
        features[f"{ax}_rms"] = rms(x)
        features[f"{ax}_skew"] = skew(x, bias=False)
        features[f"{ax}_kurtosis"] = kurtosis(x, fisher=False, bias=False)
        features[f"{ax}_range"] = np.ptp(x)
        features[f"{ax}_iqr"] = np.subtract(*np.percentile(x, [75, 25]))
        features[f"{ax}_zcr"] = zero_crossing_rate(x)

        jerk = np.diff(x) / dt
        features[f"{ax}_jerk_mean"] = np.mean(np.abs(jerk))
        features[f"{ax}_jerk_rms"] = rms(jerk)
        features[f"{ax}_jerk_peak"] = np.max(np.abs(jerk))

    features["gyr_norm_mean"] = np.mean(norm)
    features["gyr_norm_std"] = np.std(norm, ddof=1)
    features["gyr_norm_rms"] = rms(norm)
    features["gyr_norm_range"] = np.ptp(norm)
    features["gyr_norm_iqr"] = np.subtract(*np.percentile(norm, [75, 25]))
    features["gyr_norm_skew"] = skew(norm, bias=False)
    features["gyr_norm_kurtosis"] = kurtosis(norm, fisher=False, bias=False)

    jerk_norm = np.diff(norm) / dt
    features["gyr_jerk_rms_norm"] = rms(jerk_norm)
    features["gyr_jerk_peak_norm"] = np.max(np.abs(jerk_norm))

    above_noise = norm > noise_floor
    features["pct_above_noise_norm"] = np.mean(above_noise)
    run_lengths = consecutive_true_lengths(above_noise)
    features["duty_cycle_norm"] = np.sum(run_lengths) / n if len(run_lengths) else 0.0

    near_zero = norm < near_zero_thr
    nz_lengths = consecutive_true_lengths(near_zero)
    features["median_near_zero_duration_norm"] = np.median(nz_lengths) * dt if len(nz_lengths) else 0.0
    features["burst_count_norm"] = np.sum((norm[1:-1] > norm[:-2]) & (norm[1:-1] > norm[2:]))

    s = pd.Series(norm)
    features["rolling_var_mean_norm"] = s.rolling(rolling_win, min_periods=rolling_win).var().mean()
    features["moving_range_mean_norm"] = s.diff().abs().rolling(rolling_win, min_periods=rolling_win).mean().mean()

    features["corr_is_ml"] = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    features["corr_is_pa"] = np.corrcoef(data[:, 0], data[:, 2])[0, 1]
    features["corr_ml_pa"] = np.corrcoef(data[:, 1], data[:, 2])[0, 1]

    axis_energy = np.mean(data**2, axis=0)
    features["axis_dominance_ratio"] = np.max(axis_energy) / np.sum(axis_energy) if np.sum(axis_energy) > 0 else 0.0
    dominant_idx = np.argmax(axis_energy)
    features["time_gyr_norm_to_dominant_ratio"] = np.mean(norm) / (np.mean(data[:, dominant_idx]) + 1e-12)

    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            features[f"{axes[i]}_{axes[j]}_energy_ratio"] = np.sum(np.abs(data[:, i])) / (
                np.sum(np.abs(data[:, j])) + 1e-12
            )

    return features


def _extract_gyr_frequency_domain(
    df: pd.DataFrame,
    axes: Sequence[str],
    fs: float,
    lf_band: tuple[float, float],
) -> dict[str, float]:
    """Extract frequency-domain features from gyroscope data."""
    features = {}
    if df.empty or any(ax not in df.columns for ax in axes):
        return {f"{ax}_missing": np.nan for ax in axes}

    data = df[list(axes)].to_numpy(dtype=float)
    n = data.shape[0]
    if n < 8:
        return {f"{ax}_too_short": np.nan for ax in axes}

    norm = np.linalg.norm(data, axis=1)

    # 1. PSD features
    _add_gyr_psd_features(features, data, norm, axes, fs, lf_band)

    # 2. coherence features
    _add_coherence_features(features, data, axes, fs, n)

    # 3. axis relationships
    _add_axis_relation_features(features, data, norm, axes)

    return features
