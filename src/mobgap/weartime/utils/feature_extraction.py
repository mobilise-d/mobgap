"""Utility functions for IMU feature extraction from windowed data."""

from __future__ import annotations

import warnings
from collections.abc import Sequence  # noqa: TC003 - Keep annotations available for runtime inspection.
from functools import lru_cache
from math import log2

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, get_window
from scipy.stats import kurtosis, skew


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _batched_rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(values**2, axis=1))


def _batched_zero_crossing_rate(values: np.ndarray) -> np.ndarray:
    if values.shape[1] <= 1:
        return np.zeros(values.shape[0], dtype=np.float64)
    return np.mean(np.diff(np.signbit(values), axis=1) != 0, axis=1)


def _batched_iqr(values: np.ndarray) -> np.ndarray:
    has_nan = np.isnan(values).any(axis=1)
    n_values = values.shape[1]
    q25_position = (n_values - 1) * 0.25
    q75_position = (n_values - 1) * 0.75
    q25_low = int(np.floor(q25_position))
    q25_high = int(np.ceil(q25_position))
    q75_low = int(np.floor(q75_position))
    q75_high = int(np.ceil(q75_position))

    partitioned = np.partition(values, np.unique([q25_low, q25_high, q75_low, q75_high]), axis=1)
    q25_fraction = q25_position - q25_low
    q75_fraction = q75_position - q75_low
    q25 = (1 - q25_fraction) * partitioned[:, q25_low] + q25_fraction * partitioned[:, q25_high]
    q75 = (1 - q75_fraction) * partitioned[:, q75_low] + q75_fraction * partitioned[:, q75_high]
    iqr = q75 - q25
    iqr[has_nan] = np.nan
    return iqr


def _batched_spectral_centroid(frequencies: np.ndarray, power: np.ndarray) -> np.ndarray:
    total_power = power.sum(axis=1)
    weighted_power = (power * frequencies).sum(axis=1)
    return _safe_divide(weighted_power, total_power)


def _batched_spectral_entropy(power: np.ndarray) -> np.ndarray:
    total_power = power.sum(axis=1)
    psd_norm = np.zeros_like(power, dtype=np.float64)
    np.divide(power, total_power[:, None], out=psd_norm, where=total_power[:, None] > 0)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=1)
    entropy[total_power <= 0] = 0.0
    return entropy


def _batched_spectral_slope(frequencies: np.ndarray, power: np.ndarray) -> np.ndarray:
    frequency_mask = frequencies > 0
    log_frequencies = np.log(frequencies[frequency_mask])
    positive_power = power[:, frequency_mask]
    valid = positive_power > 0

    if np.all(valid):
        if len(log_frequencies) <= 2:
            return np.zeros(power.shape[0], dtype=np.float64)
        log_power = np.log(positive_power)
        count = float(len(log_frequencies))
        sum_x = np.sum(log_frequencies)
        denominator = np.sum(log_frequencies * log_frequencies) - (sum_x * sum_x / count)
        if denominator == 0:
            return np.zeros(power.shape[0], dtype=np.float64)
        sum_y = np.sum(log_power, axis=1)
        sum_xy = log_power @ log_frequencies
        return (sum_xy - (sum_x * sum_y / count)) / denominator

    log_power = np.zeros_like(positive_power, dtype=np.float64)
    np.log(positive_power, out=log_power, where=valid)

    x = log_frequencies[None, :]
    count = valid.sum(axis=1).astype(np.float64)
    sum_x = np.sum(np.where(valid, x, 0.0), axis=1)
    sum_y = np.sum(np.where(valid, log_power, 0.0), axis=1)
    sum_xx = np.sum(np.where(valid, x * x, 0.0), axis=1)
    sum_xy = np.sum(np.where(valid, x * log_power, 0.0), axis=1)

    numerator = sum_xy - _safe_divide(sum_x * sum_y, count)
    denominator = sum_xx - _safe_divide(sum_x * sum_x, count)
    slope = _safe_divide(numerator, denominator)
    slope[(count <= 2) | (denominator == 0)] = 0.0
    return slope


def _batched_top_peak_freqs(frequencies: np.ndarray, power: np.ndarray, n_peaks: int) -> np.ndarray:
    peak_freqs = np.zeros((power.shape[0], n_peaks), dtype=np.float64)
    if power.shape[1] < 3 or n_peaks <= 0:
        return peak_freqs

    peak_mask = np.zeros_like(power, dtype=bool)
    peak_mask[:, 1:-1] = (power[:, 1:-1] > power[:, :-2]) & (power[:, 1:-1] > power[:, 2:])
    peak_power = np.where(peak_mask, power, -np.inf)
    peak_order = np.argsort(peak_power, axis=1)[:, ::-1][:, :n_peaks]
    sorted_peak_power = np.take_along_axis(peak_power, peak_order, axis=1)
    sorted_peak_freqs = frequencies[peak_order]
    peak_freqs = np.where(np.isfinite(sorted_peak_power), sorted_peak_freqs, peak_freqs)

    plateau_rows = np.flatnonzero(np.any(power[:, 1:] == power[:, :-1], axis=1))
    for row in plateau_rows:
        row_peaks, _ = find_peaks(power[row])
        if len(row_peaks) == 0:
            peak_freqs[row] = 0.0
            continue
        row_peak_order = row_peaks[np.argsort(power[row, row_peaks])[::-1]][:n_peaks]
        peak_freqs[row] = 0.0
        peak_freqs[row, : len(row_peak_order)] = frequencies[row_peak_order]
    return peak_freqs


@njit(cache=True, parallel=True)
def _numba_permutation_entropy_order3(values: np.ndarray) -> np.ndarray:  # noqa: C901, PLR0912
    n_windows = values.shape[0]
    n_values = values.shape[1]
    entropy = np.zeros(n_windows, dtype=np.float64)
    if n_values < 3:
        return entropy

    n_patterns = n_values - 2
    inv_log2_6 = 1.0 / log2(6.0)
    for row in prange(n_windows):
        count_012 = 0
        count_021 = 0
        count_102 = 0
        count_120 = 0
        count_201 = 0
        count_210 = 0
        for col in range(n_patterns):
            first = values[row, col]
            second = values[row, col + 1]
            third = values[row, col + 2]
            if first <= second <= third:
                count_012 += 1
            elif first <= third < second:
                count_021 += 1
            elif second < first <= third:
                count_102 += 1
            elif second <= third < first:
                count_120 += 1
            elif third < first <= second:
                count_201 += 1
            else:
                count_210 += 1

        row_entropy = 0.0
        if count_012 > 0:
            probability = count_012 / n_patterns
            row_entropy -= probability * log2(probability)
        if count_021 > 0:
            probability = count_021 / n_patterns
            row_entropy -= probability * log2(probability)
        if count_102 > 0:
            probability = count_102 / n_patterns
            row_entropy -= probability * log2(probability)
        if count_120 > 0:
            probability = count_120 / n_patterns
            row_entropy -= probability * log2(probability)
        if count_201 > 0:
            probability = count_201 / n_patterns
            row_entropy -= probability * log2(probability)
        if count_210 > 0:
            probability = count_210 / n_patterns
            row_entropy -= probability * log2(probability)
        entropy[row] = row_entropy * inv_log2_6

    return entropy


def _batched_permutation_entropy_order3(values: np.ndarray) -> np.ndarray:
    return _numba_permutation_entropy_order3(values)


@lru_cache(maxsize=None)  # noqa: UP033 - Use the Python 3.8-compatible spelling.
def _rolling_mean_weights(n_values: int, rolling_win: int) -> np.ndarray:
    n_rolling_windows = n_values - rolling_win + 1
    return np.convolve(np.ones(n_rolling_windows, dtype=np.float64), np.ones(rolling_win, dtype=np.float64))


def _batched_moving_range_mean(values: np.ndarray, rolling_win: int) -> np.ndarray:
    diff_abs = np.abs(np.diff(values, axis=1))
    if diff_abs.shape[1] < rolling_win:
        return np.full(values.shape[0], np.nan)

    weights = _rolling_mean_weights(diff_abs.shape[1], rolling_win)
    return (diff_abs @ weights) / (rolling_win * (diff_abs.shape[1] - rolling_win + 1))


def _batched_rolling_var_mean_std(values: np.ndarray, rolling_win: int) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[1] < rolling_win:
        missing = np.full(values.shape[0], np.nan)
        return missing, missing.copy()

    padded = np.pad(values, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
    cumsum = np.cumsum(padded, axis=1)
    cumsum_sq = np.cumsum(padded * padded, axis=1)
    rolling_sum = cumsum[:, rolling_win:] - cumsum[:, :-rolling_win]
    rolling_sum_sq = cumsum_sq[:, rolling_win:] - cumsum_sq[:, :-rolling_win]
    rolling_var = (rolling_sum_sq - rolling_sum * rolling_sum / rolling_win) / (rolling_win - 1)
    rolling_var_mean = np.mean(rolling_var, axis=1)
    if rolling_var.shape[1] <= 1:
        return rolling_var_mean, np.full(values.shape[0], np.nan)
    return rolling_var_mean, np.std(rolling_var, axis=1, ddof=1)


@njit(cache=True, parallel=True)
def _numba_true_run_median_max(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_windows = mask.shape[0]
    n_values = mask.shape[1]
    medians = np.zeros(n_windows, dtype=np.float64)
    maximums = np.zeros(n_windows, dtype=np.float64)
    for row in prange(n_windows):
        run_lengths = np.empty(n_values, dtype=np.int64)
        n_runs = 0
        run_length = 0
        for col in range(n_values):
            if mask[row, col]:
                run_length += 1
            elif run_length:
                run_lengths[n_runs] = run_length
                n_runs += 1
                run_length = 0
        if run_length:
            run_lengths[n_runs] = run_length
            n_runs += 1

        if n_runs == 0:
            continue

        for i in range(1, n_runs):
            value = run_lengths[i]
            j = i - 1
            while j >= 0 and run_lengths[j] > value:
                run_lengths[j + 1] = run_lengths[j]
                j -= 1
            run_lengths[j + 1] = value

        maximums[row] = run_lengths[n_runs - 1]
        midpoint = n_runs // 2
        if n_runs % 2:
            medians[row] = run_lengths[midpoint]
        else:
            medians[row] = (run_lengths[midpoint - 1] + run_lengths[midpoint]) / 2.0
    return medians, maximums


def _batched_true_run_median_max(mask: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    medians, maximums = _numba_true_run_median_max(mask)
    return medians * dt, maximums * dt


def _batched_corrcoef(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_centered = first - np.mean(first, axis=1, keepdims=True)
    second_centered = second - np.mean(second, axis=1, keepdims=True)
    numerator = np.sum(first_centered * second_centered, axis=1)
    denominator = np.sqrt(
        np.sum(first_centered * first_centered, axis=1) * np.sum(second_centered * second_centered, axis=1)
    )
    result = np.full(first.shape[0], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _batched_coherence_mean(first: np.ndarray, second: np.ndarray, fs: float) -> np.ndarray:
    frequencies, window, _ = _welch_frequencies_and_window(first.shape[1], fs)
    first_spectrum = rfft((first - np.mean(first, axis=1, keepdims=True)) * window[None, :], axis=1, workers=-1)
    second_spectrum = rfft((second - np.mean(second, axis=1, keepdims=True)) * window[None, :], axis=1, workers=-1)
    numerator = np.abs(first_spectrum.conj() * second_spectrum) ** 2
    denominator = np.abs(first_spectrum) ** 2 * np.abs(second_spectrum) ** 2
    coherence = np.full((first.shape[0], len(frequencies)), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=coherence, where=denominator != 0)
    return np.mean(coherence, axis=1)


@lru_cache(maxsize=None)  # noqa: UP033 - Use the Python 3.8-compatible spelling.
def _welch_frequencies_and_window(n_values: int, fs: float) -> tuple[np.ndarray, np.ndarray, float]:
    window = get_window("hann", n_values)
    scale = 1.0 / (fs * np.sum(window * window))
    return rfftfreq(n_values, 1.0 / fs), window, scale


def _batched_welch(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n_values = values.shape[1]
    frequencies, window, scale = _welch_frequencies_and_window(n_values, fs)
    detrended = values - np.mean(values, axis=1, keepdims=True)
    spectrum = rfft(detrended * window[None, :], axis=1, workers=-1)
    power = np.real(spectrum * spectrum.conj()) * scale
    if n_values % 2 == 0:
        power[:, 1:-1] *= 2.0
    else:
        power[:, 1:] *= 2.0
    return frequencies, power


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


def _windowed_sensor_arrays(
    data: pd.DataFrame | np.ndarray,
    starts: np.ndarray,
    window_samples: int,
    acc_axes: tuple[str, str, str],
    gyr_axes: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build window views from six columns ordered as acceleration, then gyroscope."""
    if isinstance(data, pd.DataFrame):
        sensor_values = data.loc[:, [*acc_axes, *gyr_axes]].to_numpy(dtype=float, copy=False)
    else:
        sensor_values = np.asarray(data, dtype=float)

    step = int(starts[1] - starts[0]) if len(starts) > 1 else 1
    if (
        starts[0] >= 0
        and starts[-1] + window_samples <= len(sensor_values)
        and step > 0
        and np.all(np.diff(starts) == step)
    ):
        windows = np.lib.stride_tricks.sliding_window_view(sensor_values, window_samples, axis=0)
        windows = windows[starts[0] : starts[-1] + 1 : step].transpose(0, 2, 1)
    else:
        windows = sensor_values[starts[:, None] + np.arange(window_samples)]
    return windows[:, :, :3], windows[:, :, 3:]


def extract_features_90pct_batched(  # noqa: PLR0915
    df: pd.DataFrame | np.ndarray,
    window_start_end: np.ndarray,
    acc_axes: tuple[str, str, str] = ("acc_is", "acc_ml", "acc_pa"),
    gyr_axes: tuple[str, str, str] = ("gyr_is", "gyr_ml", "gyr_pa"),
    fs: float = 100.0,
    dt: float = 0.01,
    lf_band: tuple[float, float] = (0.0, 0.5),
    rolling_win: int = 10,
    feature_names: Sequence[str] = (),
) -> pd.DataFrame:
    """Extract the 90% XGBoost feature set for a batch of fixed-size windows."""
    if len(window_start_end) == 0:
        return pd.DataFrame(columns=list(feature_names or FEATURE_ORDER_90PCT))

    starts = window_start_end[:, 0].astype(np.int64, copy=False)
    window_samples = int(window_start_end[0, 1] - window_start_end[0, 0])
    if not np.all(window_start_end[:, 1] - window_start_end[:, 0] == window_samples):
        raise ValueError("Batched feature extraction requires fixed-size windows.")

    expected_samples = int(5.0 * fs)
    if window_samples != expected_samples:
        warnings.warn(
            f"Input window duration is {window_samples / fs:.2f}s. "
            f"Model was trained on 5.0s windows. Performance may be affected.",
            UserWarning,
            stacklevel=2,
        )

    acc_data, gyr_data = _windowed_sensor_arrays(df, starts, window_samples, acc_axes, gyr_axes)

    acc_is = acc_data[:, :, 0]
    acc_ml = acc_data[:, :, 1]
    acc_pa = acc_data[:, :, 2]
    gyr_is = gyr_data[:, :, 0]
    gyr_ml = gyr_data[:, :, 1]
    gyr_pa = gyr_data[:, :, 2]
    acc_norm = np.sqrt(acc_is * acc_is + acc_ml * acc_ml + acc_pa * acc_pa)
    gyr_norm = np.sqrt(gyr_is * gyr_is + gyr_ml * gyr_ml + gyr_pa * gyr_pa)

    f_acc_is, pxx_acc_is = _batched_welch(acc_is, fs)
    f_acc_ml, pxx_acc_ml = _batched_welch(acc_ml, fs)
    f_acc_pa, pxx_acc_pa = _batched_welch(acc_pa, fs)
    f_acc_norm, pxx_acc_norm = _batched_welch(acc_norm, fs)
    f_gyr_is, pxx_gyr_is = _batched_welch(gyr_is, fs)
    f_gyr_ml, pxx_gyr_ml = _batched_welch(gyr_ml, fs)
    _, pxx_gyr_pa = _batched_welch(gyr_pa, fs)
    f_gyr_norm, pxx_gyr_norm = _batched_welch(gyr_norm, fs)

    acc_fft = np.abs(rfft(acc_data, axis=1, workers=-1)).sum(axis=1)
    gyr_fft = np.abs(rfft(gyr_data, axis=1, workers=-1)).sum(axis=1)
    acc_abs_sum = np.abs(acc_data).sum(axis=1)
    gyr_abs_sum = np.abs(gyr_data).sum(axis=1)

    jerk_acc_is = np.diff(acc_is, axis=1) / dt
    jerk_acc_norm = np.diff(acc_norm, axis=1) / dt
    jerk_gyr_is = np.diff(gyr_is, axis=1) / dt
    jerk_gyr_ml = np.diff(gyr_ml, axis=1) / dt
    jerk_gyr_pa = np.diff(gyr_pa, axis=1) / dt

    acc_axis_energy = np.mean(acc_data**2, axis=1)
    acc_dominant_idx = np.argmax(acc_axis_energy, axis=1)
    acc_axis_mean = np.mean(acc_data, axis=1)
    gyr_axis_energy = np.mean(gyr_data**2, axis=1)
    gyr_dominant_idx = np.argmax(gyr_axis_energy, axis=1)
    gyr_axis_mean = np.mean(gyr_data, axis=1)

    acc_pa_peak_freqs = _batched_top_peak_freqs(f_acc_pa, pxx_acc_pa, 3)
    acc_norm_peak_freqs = _batched_top_peak_freqs(f_acc_norm, pxx_acc_norm, 2)
    gyr_ml_peak_freqs = _batched_top_peak_freqs(f_gyr_ml, pxx_gyr_ml, 1)

    total_acc_pa = pxx_acc_pa.sum(axis=1)
    lf_mask = (f_gyr_ml >= lf_band[0]) & (f_gyr_ml <= lf_band[1])
    hf_mask = f_gyr_ml > lf_band[1]
    gyr_ml_lf_power = pxx_gyr_ml[:, lf_mask].sum(axis=1)
    gyr_ml_hf_power = pxx_gyr_ml[:, hf_mask].sum(axis=1)

    features = {
        "acc_pa_std": np.std(acc_pa, axis=1, ddof=1),
        "gyr_ml_spectral_centroid": _batched_spectral_centroid(f_gyr_ml, pxx_gyr_ml),
        "gyr_is_spectral_centroid": _batched_spectral_centroid(f_gyr_is, pxx_gyr_is),
        "time_acc_norm_to_dominant_ratio": np.mean(acc_norm, axis=1)
        / (acc_axis_mean[np.arange(len(acc_axis_mean)), acc_dominant_idx] + 1e-12),
        "gyr_ml_lf_power": gyr_ml_lf_power,
        "gyr_ml_spectral_slope": _batched_spectral_slope(f_gyr_ml, pxx_gyr_ml),
        "acc_pa_spectral_centroid": _batched_spectral_centroid(f_acc_pa, pxx_acc_pa),
        "gyr_is_spectral_slope": _batched_spectral_slope(f_gyr_is, pxx_gyr_is),
        "gyr_pa_mean": np.mean(gyr_pa, axis=1),
        "acc_is_jerk_mean": np.mean(np.abs(jerk_acc_is), axis=1),
        "gyr_is_std": np.std(gyr_is, axis=1, ddof=1),
        "gyr_is_gyr_pa_energy_ratio": gyr_abs_sum[:, 0] / (gyr_abs_sum[:, 2] + 1e-12),
        "gyr_ml_mean": np.mean(gyr_ml, axis=1),
        "acc_is_mean": np.mean(acc_is, axis=1),
        "axis_dominance_ratio": _safe_divide(np.max(gyr_axis_energy, axis=1), np.sum(gyr_axis_energy, axis=1)),
        "gyr_pa_rms": _batched_rms(gyr_pa),
        "gyr_ml_iqr": _batched_iqr(gyr_ml),
        "acc_is_min": np.min(acc_is, axis=1),
        "acc_pa_rms": _batched_rms(acc_pa),
        "acc_is_acc_pa_psd_ratio": acc_fft[:, 0] / (acc_fft[:, 2] + 1e-12),
        "acc_is_median": np.median(acc_is, axis=1),
        "acc_jerk_std_norm": np.std(jerk_acc_norm, axis=1, ddof=1),
        "gyr_is_mean": np.mean(gyr_is, axis=1),
        "gyr_ml_lf_hf_ratio": _safe_divide(gyr_ml_lf_power, gyr_ml_hf_power),
        "acc_is_max": np.max(acc_is, axis=1),
        "acc_norm_spectral_centroid": _batched_spectral_centroid(f_acc_norm, pxx_acc_norm),
        "gyr_ml_gyr_pa_energy_ratio": gyr_abs_sum[:, 1] / (gyr_abs_sum[:, 2] + 1e-12),
        "acc_pa_median": np.median(acc_pa, axis=1),
        "acc_pa_second_peak_freq": acc_pa_peak_freqs[:, 1],
        "acc_is_spectral_slope": _batched_spectral_slope(f_acc_is, pxx_acc_is),
        "acc_is_acc_ml_psd_ratio": acc_fft[:, 0] / (acc_fft[:, 1] + 1e-12),
        "gyr_is_perm_entropy": _batched_permutation_entropy_order3(gyr_is),
        "gyr_is_gyr_pa_psd_ratio": gyr_fft[:, 0] / (gyr_fft[:, 2] + 1e-12),
        "moving_range_mean_norm": _batched_moving_range_mean(gyr_norm, rolling_win),
        "acc_norm_dom_freq": acc_norm_peak_freqs[:, 0],
        "gyr_pa_jerk_mean": np.mean(np.abs(jerk_gyr_pa), axis=1),
        "acc_is_perm_entropy": _batched_permutation_entropy_order3(acc_is),
        "acc_ml_mean": np.mean(acc_ml, axis=1),
        "gyr_is_jerk_mean": np.mean(np.abs(jerk_gyr_is), axis=1),
        "acc_jerk_rms_norm": _batched_rms(jerk_acc_norm),
        "acc_pa_dom_freq": acc_pa_peak_freqs[:, 0],
        "acc_norm_median": np.median(acc_norm, axis=1),
        "gyr_is_kurtosis": kurtosis(gyr_is, axis=1, fisher=False, bias=False),
        "acc_ml_median": np.median(acc_ml, axis=1),
        "gyr_pa_iqr": _batched_iqr(gyr_pa),
        "acc_ml_perm_entropy": _batched_permutation_entropy_order3(acc_ml),
        "acc_pa_mean": np.mean(acc_pa, axis=1),
        "acc_jerk_peak_norm": np.max(np.abs(jerk_acc_norm), axis=1),
        "acc_ml_acc_pa_energy_ratio": acc_abs_sum[:, 1] / (acc_abs_sum[:, 2] + 1e-12),
        "gyr_norm_spectral_slope": _batched_spectral_slope(f_gyr_norm, pxx_gyr_norm),
        "gyr_is_gyr_ml_energy_ratio": gyr_abs_sum[:, 0] / (gyr_abs_sum[:, 1] + 1e-12),
        "acc_pa_max": np.max(acc_pa, axis=1),
        "gyr_is_zcr": _batched_zero_crossing_rate(gyr_is),
        "acc_pa_perm_entropy": _batched_permutation_entropy_order3(acc_pa),
        "acc_is_acc_ml_energy_ratio": acc_abs_sum[:, 0] / (acc_abs_sum[:, 1] + 1e-12),
        "gyr_is_jerk_rms": _batched_rms(jerk_gyr_is),
        "acc_norm_second_peak_freq": acc_norm_peak_freqs[:, 1],
        "acc_is_jerk_std": np.std(jerk_acc_is, axis=1, ddof=1),
        "gyr_norm_skew": skew(gyr_norm, axis=1, bias=False),
        "acc_ml_spectral_slope": _batched_spectral_slope(f_acc_ml, pxx_acc_ml),
        "acc_pa_min": np.min(acc_pa, axis=1),
        "acc_norm_mean": np.mean(acc_norm, axis=1),
        "acc_pa_psd_total": total_acc_pa,
        "acc_pa_third_peak_freq": acc_pa_peak_freqs[:, 2],
        "burst_count_norm": np.sum(
            (gyr_norm[:, 1:-1] > gyr_norm[:, :-2]) & (gyr_norm[:, 1:-1] > gyr_norm[:, 2:]),
            axis=1,
        ),
        "acc_is_zcr": _batched_zero_crossing_rate(acc_is),
        "acc_norm_rms": _batched_rms(acc_norm),
        "gyr_pa_zcr": _batched_zero_crossing_rate(gyr_pa),
        "acc_ml_rms": _batched_rms(acc_ml),
        "acc_pa_spectral_slope": _batched_spectral_slope(f_acc_pa, pxx_acc_pa),
        "acc_norm_spectral_slope": _batched_spectral_slope(f_acc_norm, pxx_acc_norm),
        "gyr_ml_jerk_rms": _batched_rms(jerk_gyr_ml),
        "gyr_ml_gyr_pa_psd_ratio": gyr_fft[:, 1] / (gyr_fft[:, 2] + 1e-12),
        "gyr_pa_spectral_entropy": _batched_spectral_entropy(pxx_gyr_pa),
        "time_gyr_norm_to_dominant_ratio": np.mean(gyr_norm, axis=1)
        / (gyr_axis_mean[np.arange(len(gyr_axis_mean)), gyr_dominant_idx] + 1e-12),
        "acc_ml_acc_pa_psd_ratio": acc_fft[:, 1] / (acc_fft[:, 2] + 1e-12),
        "gyr_ml_dom_freq": gyr_ml_peak_freqs[:, 0],
        "gyr_is_rms": _batched_rms(gyr_is),
        "gyr_norm_perm_entropy": _batched_permutation_entropy_order3(gyr_norm),
    }

    feature_names = feature_names or FEATURE_ORDER_90PCT
    return pd.DataFrame({feature_name: features[feature_name] for feature_name in feature_names})


def _batched_psd_features(
    values: np.ndarray,
    prefix: str,
    fs: float,
    lf_band: tuple[float, float],
) -> dict[str, np.ndarray]:
    frequencies, power = _batched_welch(values, fs)
    total_power = power.sum(axis=1)
    lf_mask = (frequencies >= lf_band[0]) & (frequencies <= lf_band[1])
    hf_mask = frequencies > lf_band[1]
    lf_power = power[:, lf_mask].sum(axis=1)
    hf_power = power[:, hf_mask].sum(axis=1)
    peak_freqs = _batched_top_peak_freqs(frequencies, power, 3)
    return {
        f"{prefix}_psd_total": total_power,
        f"{prefix}_lf_power": lf_power,
        f"{prefix}_lf_hf_ratio": _safe_divide(lf_power, hf_power),
        f"{prefix}_dom_freq": peak_freqs[:, 0],
        f"{prefix}_second_peak_freq": peak_freqs[:, 1],
        f"{prefix}_third_peak_freq": peak_freqs[:, 2],
        f"{prefix}_spectral_centroid": _batched_spectral_centroid(frequencies, power),
        f"{prefix}_spectral_entropy": _batched_spectral_entropy(power),
        f"{prefix}_spectral_skewness": skew(power, axis=1, bias=False),
        f"{prefix}_spectral_kurtosis": kurtosis(power, axis=1, fisher=False, bias=False),
        f"{prefix}_spectral_slope": _batched_spectral_slope(frequencies, power),
        f"{prefix}_perm_entropy": _batched_permutation_entropy_order3(values),
    }


def extract_full_features_batched(  # noqa: PLR0915
    df: pd.DataFrame | np.ndarray,
    window_start_end: np.ndarray,
    acc_axes: tuple[str, str, str] = ("acc_is", "acc_ml", "acc_pa"),
    gyr_axes: tuple[str, str, str] = ("gyr_is", "gyr_ml", "gyr_pa"),
    fs: float = 100.0,
    dt: float = 0.01,
    lf_band: tuple[float, float] = (0.0, 0.5),
    noise_floor: float = 0.05,
    near_zero_thr: float = 0.02,
    rolling_win: int = 10,
    feature_names: Sequence[str] = (),
) -> pd.DataFrame:
    """Extract the full XGBoost feature set for a batch of fixed-size windows."""
    if len(window_start_end) == 0:
        return pd.DataFrame(columns=list(feature_names or FULL_FEATURE_ORDER))

    starts = window_start_end[:, 0].astype(np.int64, copy=False)
    window_samples = int(window_start_end[0, 1] - window_start_end[0, 0])
    if not np.all(window_start_end[:, 1] - window_start_end[:, 0] == window_samples):
        raise ValueError("Batched feature extraction requires fixed-size windows.")

    expected_samples = int(5.0 * fs)
    if window_samples != expected_samples:
        warnings.warn(
            f"Input window duration is {window_samples / fs:.2f}s. "
            f"Model was trained on 5.0s windows. Performance may be affected.",
            UserWarning,
            stacklevel=2,
        )

    acc_data, gyr_data = _windowed_sensor_arrays(df, starts, window_samples, acc_axes, gyr_axes)

    acc_is = acc_data[:, :, 0]
    acc_ml = acc_data[:, :, 1]
    acc_pa = acc_data[:, :, 2]
    gyr_is = gyr_data[:, :, 0]
    gyr_ml = gyr_data[:, :, 1]
    gyr_pa = gyr_data[:, :, 2]
    acc_norm = np.sqrt(acc_is * acc_is + acc_ml * acc_ml + acc_pa * acc_pa)
    gyr_norm = np.sqrt(gyr_is * gyr_is + gyr_ml * gyr_ml + gyr_pa * gyr_pa)

    jerk_acc = np.diff(acc_data, axis=1) / dt
    jerk_gyr = np.diff(gyr_data, axis=1) / dt
    jerk_acc_norm = np.diff(acc_norm, axis=1) / dt
    jerk_gyr_norm = np.diff(gyr_norm, axis=1) / dt

    acc_abs_sum = np.abs(acc_data).sum(axis=1)
    gyr_abs_sum = np.abs(gyr_data).sum(axis=1)
    acc_fft = np.abs(rfft(acc_data, axis=1, workers=-1)).sum(axis=1)
    gyr_fft = np.abs(rfft(gyr_data, axis=1, workers=-1)).sum(axis=1)

    acc_axis_energy = np.mean(acc_data**2, axis=1)
    gyr_axis_energy = np.mean(gyr_data**2, axis=1)
    acc_dominant_idx = np.argmax(acc_axis_energy, axis=1)
    gyr_dominant_idx = np.argmax(gyr_axis_energy, axis=1)
    acc_axis_mean = np.mean(acc_data, axis=1)
    gyr_axis_mean = np.mean(gyr_data, axis=1)
    rows = np.arange(len(acc_data))

    acc_near_zero_median, acc_near_zero_max = _batched_true_run_median_max(acc_norm < near_zero_thr, dt)
    gyr_near_zero_median, _ = _batched_true_run_median_max(gyr_norm < near_zero_thr, dt)
    acc_rolling_var_mean, acc_rolling_var_std = _batched_rolling_var_mean_std(acc_norm, rolling_win)
    gyr_rolling_var_mean, _ = _batched_rolling_var_mean_std(gyr_norm, rolling_win)

    features: dict[str, np.ndarray] = {
        "gyr_norm_mean": np.mean(gyr_norm, axis=1),
        "gyr_norm_std": np.std(gyr_norm, axis=1, ddof=1),
        "gyr_norm_rms": _batched_rms(gyr_norm),
        "gyr_norm_range": np.ptp(gyr_norm, axis=1),
        "gyr_norm_iqr": _batched_iqr(gyr_norm),
        "gyr_norm_skew": skew(gyr_norm, axis=1, bias=False),
        "gyr_norm_kurtosis": kurtosis(gyr_norm, axis=1, fisher=False, bias=False),
        "gyr_jerk_rms_norm": _batched_rms(jerk_gyr_norm),
        "gyr_jerk_peak_norm": np.max(np.abs(jerk_gyr_norm), axis=1),
        "pct_above_noise_norm": np.mean(gyr_norm > noise_floor, axis=1),
        "duty_cycle_norm": np.mean(gyr_norm > noise_floor, axis=1),
        "median_near_zero_duration_norm": gyr_near_zero_median,
        "burst_count_norm": np.sum(
            (gyr_norm[:, 1:-1] > gyr_norm[:, :-2]) & (gyr_norm[:, 1:-1] > gyr_norm[:, 2:]),
            axis=1,
        ),
        "rolling_var_mean_norm": gyr_rolling_var_mean,
        "moving_range_mean_norm": _batched_moving_range_mean(gyr_norm, rolling_win),
        "corr_is_ml": _batched_corrcoef(gyr_is, gyr_ml),
        "corr_is_pa": _batched_corrcoef(gyr_is, gyr_pa),
        "corr_ml_pa": _batched_corrcoef(gyr_ml, gyr_pa),
        "axis_dominance_ratio": _safe_divide(np.max(gyr_axis_energy, axis=1), np.sum(gyr_axis_energy, axis=1)),
        "time_gyr_norm_to_dominant_ratio": np.mean(gyr_norm, axis=1) / (gyr_axis_mean[rows, gyr_dominant_idx] + 1e-12),
        "freq_gyr_norm_to_dominant_ratio": np.mean(gyr_norm, axis=1) / (gyr_axis_mean[rows, gyr_dominant_idx] + 1e-12),
        "gyr_is_gyr_ml_energy_ratio": gyr_abs_sum[:, 0] / (gyr_abs_sum[:, 1] + 1e-12),
        "gyr_is_gyr_pa_energy_ratio": gyr_abs_sum[:, 0] / (gyr_abs_sum[:, 2] + 1e-12),
        "gyr_ml_gyr_pa_energy_ratio": gyr_abs_sum[:, 1] / (gyr_abs_sum[:, 2] + 1e-12),
        "gyr_is_gyr_ml_psd_ratio": gyr_fft[:, 0] / (gyr_fft[:, 1] + 1e-12),
        "gyr_is_gyr_pa_psd_ratio": gyr_fft[:, 0] / (gyr_fft[:, 2] + 1e-12),
        "gyr_ml_gyr_pa_psd_ratio": gyr_fft[:, 1] / (gyr_fft[:, 2] + 1e-12),
        "acc_norm_mean": np.mean(acc_norm, axis=1),
        "acc_norm_std": np.std(acc_norm, axis=1, ddof=1),
        "acc_norm_rms": _batched_rms(acc_norm),
        "acc_norm_median": np.median(acc_norm, axis=1),
        "acc_norm_min": np.min(acc_norm, axis=1),
        "acc_norm_max": np.max(acc_norm, axis=1),
        "acc_norm_range": np.ptp(acc_norm, axis=1),
        "acc_norm_iqr": _batched_iqr(acc_norm),
        "acc_norm_skew": skew(acc_norm, axis=1, bias=False),
        "acc_norm_kurtosis": kurtosis(acc_norm, axis=1, fisher=False, bias=False),
        "acc_jerk_rms_norm": _batched_rms(jerk_acc_norm),
        "acc_jerk_peak_norm": np.max(np.abs(jerk_acc_norm), axis=1),
        "acc_jerk_std_norm": np.std(jerk_acc_norm, axis=1, ddof=1),
        "max_near_zero_duration_norm": acc_near_zero_max,
        "rolling_var_std_norm": acc_rolling_var_std,
        "time_acc_norm_to_dominant_ratio": np.mean(acc_norm, axis=1) / (acc_axis_mean[rows, acc_dominant_idx] + 1e-12),
        "freq_acc_norm_to_dominant_ratio": np.mean(acc_norm, axis=1) / (acc_axis_mean[rows, acc_dominant_idx] + 1e-12),
        "acc_is_acc_ml_energy_ratio": acc_abs_sum[:, 0] / (acc_abs_sum[:, 1] + 1e-12),
        "acc_is_acc_pa_energy_ratio": acc_abs_sum[:, 0] / (acc_abs_sum[:, 2] + 1e-12),
        "acc_ml_acc_pa_energy_ratio": acc_abs_sum[:, 1] / (acc_abs_sum[:, 2] + 1e-12),
        "acc_is_acc_ml_psd_ratio": acc_fft[:, 0] / (acc_fft[:, 1] + 1e-12),
        "acc_is_acc_pa_psd_ratio": acc_fft[:, 0] / (acc_fft[:, 2] + 1e-12),
        "acc_ml_acc_pa_psd_ratio": acc_fft[:, 1] / (acc_fft[:, 2] + 1e-12),
        "gyr_is_gyr_ml_coherence_mean": _batched_coherence_mean(gyr_is, gyr_ml, fs),
        "gyr_is_gyr_pa_coherence_mean": _batched_coherence_mean(gyr_is, gyr_pa, fs),
        "gyr_ml_gyr_pa_coherence_mean": _batched_coherence_mean(gyr_ml, gyr_pa, fs),
        "acc_is_acc_ml_coherence_mean": _batched_coherence_mean(acc_is, acc_ml, fs),
        "acc_is_acc_pa_coherence_mean": _batched_coherence_mean(acc_is, acc_pa, fs),
        "acc_ml_acc_pa_coherence_mean": _batched_coherence_mean(acc_ml, acc_pa, fs),
    }

    for axis_index, axis_name in enumerate(gyr_axes):
        axis_values = gyr_data[:, :, axis_index]
        axis_jerk = jerk_gyr[:, :, axis_index]
        features.update(
            {
                f"{axis_name}_mean": np.mean(axis_values, axis=1),
                f"{axis_name}_std": np.std(axis_values, axis=1, ddof=1),
                f"{axis_name}_rms": _batched_rms(axis_values),
                f"{axis_name}_skew": skew(axis_values, axis=1, bias=False),
                f"{axis_name}_kurtosis": kurtosis(axis_values, axis=1, fisher=False, bias=False),
                f"{axis_name}_range": np.ptp(axis_values, axis=1),
                f"{axis_name}_iqr": _batched_iqr(axis_values),
                f"{axis_name}_zcr": _batched_zero_crossing_rate(axis_values),
                f"{axis_name}_jerk_mean": np.mean(np.abs(axis_jerk), axis=1),
                f"{axis_name}_jerk_rms": _batched_rms(axis_jerk),
                f"{axis_name}_jerk_peak": np.max(np.abs(axis_jerk), axis=1),
            }
        )
        features.update(_batched_psd_features(axis_values, axis_name, fs, lf_band))

    for axis_index, axis_name in enumerate(acc_axes):
        axis_values = acc_data[:, :, axis_index]
        axis_jerk = jerk_acc[:, :, axis_index]
        features.update(
            {
                f"{axis_name}_mean": np.mean(axis_values, axis=1),
                f"{axis_name}_std": np.std(axis_values, axis=1, ddof=1),
                f"{axis_name}_rms": _batched_rms(axis_values),
                f"{axis_name}_median": np.median(axis_values, axis=1),
                f"{axis_name}_min": np.min(axis_values, axis=1),
                f"{axis_name}_max": np.max(axis_values, axis=1),
                f"{axis_name}_range": np.ptp(axis_values, axis=1),
                f"{axis_name}_iqr": _batched_iqr(axis_values),
                f"{axis_name}_skew": skew(axis_values, axis=1, bias=False),
                f"{axis_name}_kurtosis": kurtosis(axis_values, axis=1, fisher=False, bias=False),
                f"{axis_name}_zcr": _batched_zero_crossing_rate(axis_values),
                f"{axis_name}_jerk_mean": np.mean(np.abs(axis_jerk), axis=1),
                f"{axis_name}_jerk_rms": _batched_rms(axis_jerk),
                f"{axis_name}_jerk_peak": np.max(np.abs(axis_jerk), axis=1),
                f"{axis_name}_jerk_std": np.std(axis_jerk, axis=1, ddof=1),
            }
        )
        features.update(_batched_psd_features(axis_values, axis_name, fs, lf_band))

    features.update(_batched_psd_features(gyr_norm, "gyr_norm", fs, lf_band))
    features.update(_batched_psd_features(acc_norm, "acc_norm", fs, lf_band))

    # Keep the historical full-feature semantics where these shared names are produced by the gyroscope path.
    del acc_rolling_var_mean
    del acc_near_zero_median

    feature_names = feature_names or FULL_FEATURE_ORDER
    return pd.DataFrame({feature_name: features[feature_name] for feature_name in feature_names})


def extract_features_batched(
    df: pd.DataFrame | np.ndarray,
    window_start_end: np.ndarray,
    acc_axes: tuple[str, str, str] = ("acc_is", "acc_ml", "acc_pa"),
    gyr_axes: tuple[str, str, str] = ("gyr_is", "gyr_ml", "gyr_pa"),
    fs: float = 100.0,
    dt: float = 0.01,
    lf_band: tuple[float, float] = (0.0, 0.5),
    rolling_win: int = 10,
    feature_names: Sequence[str] = (),
    version: str = "lightweight",
) -> pd.DataFrame:
    """Extract XGBoost wear-time features using only batched feature calculators."""
    if version == "full":
        return extract_full_features_batched(
            df,
            window_start_end,
            acc_axes=acc_axes,
            gyr_axes=gyr_axes,
            fs=fs,
            dt=dt,
            lf_band=lf_band,
            rolling_win=rolling_win,
            feature_names=feature_names,
        )

    requested_features = tuple(feature_names or FEATURE_ORDER_90PCT)
    unsupported_features = set(requested_features) - set(FEATURE_ORDER_90PCT)
    if unsupported_features:
        raise ValueError(
            "The lightweight XGBoost feature set only supports `FEATURE_ORDER_90PCT` features. "
            f"Unsupported features: {sorted(unsupported_features)}"
        )
    return extract_features_90pct_batched(
        df,
        window_start_end,
        acc_axes=acc_axes,
        gyr_axes=gyr_axes,
        fs=fs,
        dt=dt,
        lf_band=lf_band,
        rolling_win=rolling_win,
        feature_names=requested_features,
    )
