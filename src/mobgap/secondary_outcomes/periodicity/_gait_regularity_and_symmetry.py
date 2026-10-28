"""GaitRegularityAndSymmetry calculation for one prepared window."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks
from typing_extensions import Self, Unpack

from mobgap._utils_internal.misc import timed_action_method
from mobgap.secondary_outcomes.base import BaseSecondaryOutcomeCalculator, base_secondary_outcome_docfiller


@base_secondary_outcome_docfiller
class GaitRegularityAndSymmetry(BaseSecondaryOutcomeCalculator):
    """Compute step/stride regularity and symmetry metrics from accelerations.

    This variant aims to reproduce the original MATLAB regularity/symmetry implementation as closely as possible.
    In particular, its smoothing is intended to match MATLAB's ``smooth(y, span, 'moving')`` exactly: reduce even
    spans by one and use shrinking symmetric windows at the edges. Full numerical equivalence to the original
    MATLAB calculator has not been established: the original ``CorrectPeaks`` helper was unavailable, and Python
    uses the peak-correction, lag-selection and partial-result rules described below.
    :class:`~mobgap.secondary_outcomes.GaitRegularityAndSymmetryTruncatedMean` instead uses standard pandas
    rolling-mean semantics, avoiding the custom emulation of MATLAB smoothing.

    Step regularity
        Expresses the regularity of the acceleration signal between neighboring steps. Values close to 1 indicate
        greater regularity of the gait pattern. For the mediolateral axis, the absolute
        value of the first negative peak is reported. Implemented from [1]_.
    Stride regularity
        Expresses the regularity of the acceleration signal between neighboring strides. Values close to 1 indicate
        greater regularity of the gait pattern. Implemented from [1]_.

    This class outputs the step and stride regularity for all available acceleration signals in the principal
    directions.
    The parameters below (Asymmetry_MN, Symmetry_K, Asymmetry_G) are calculated for all axes, but only vertical-axis
    components are included in the output.

    Asymmetry_MN
        Step symmetry is defined as the ratio of step regularity to stride regularity. Closeness of the symmetry to 1
        reflects symmetry. Asymmetry_MN is defined as how close this ratio is to 1 ``abs(1 - symmetry)``.
        Implemented from [1]_.

    Symmetry_K
        It is defined as the relative symmetry between step and stride regularity. It is calculated as the
        absolute difference between the two values divided by their mean. Value of zero means perfect symmetry and
        larger values depicting greater levels of step asymmetry in the accelerometer waveform. Implemented from [2]_.

    Asymmetry_G
        The exported IS result uses the linear formula ``(stride regularity-step regularity)/2`` from [3]_.
        Zero indicates equal step and stride regularity. Positive values indicate higher stride regularity,
        while negative values indicate the opposite. Other asymmetry axes are not exported.

    Notes
    -----
    The cited references define autocovariance-based regularity and the derived symmetry formulas. Additional
    implementation choices smooth the normalized autocovariance over approximately 0.2 s for IS/PA and 0.1 s for ML,
    detect peaks with a minimum separation of ``floor(sampling_rate_hz / 4) + 1`` samples, and adjust them against the
    unsmoothed curve. IS/PA use the first two nonnegative candidates; ML uses a negative trough and a positive peak,
    with lower lag bounds guided by the IS step lag or PA fallback. No ICs, cadence or other outcome estimates are used.
    For two or more candidates, correction uses a window derived from 20 percent of median candidate spacing,
    extending farther backward than forward, excludes boundary candidates and merges nearby corrected peaks.
    With fewer than two candidates, correction is skipped and smoothed heights are retained. These selection and
    correction details are implementation choices, not procedures established by the cited studies.
    Normalization of unbiased autocovariance does not guarantee values within [-1, 1]; outputs are not clipped.
    Short or irregular bouts can yield ambiguous periods, and ratio-based asymmetry is sensitive to small denominators.

    Autocovariance smoothing uses an odd centered moving average with shrinking symmetric edge windows.
    An even nominal span is reduced by one. For the variant using the nominal span and truncated edge windows,
    see :class:`~mobgap.secondary_outcomes.GaitRegularityAndSymmetryTruncatedMean`.

    Other Parameters
    ----------------
    %(data_param)s
    %(sampling_rate_param)s

    Attributes
    ----------
    %(signal_based_parameters_)s
    %(perf_)s

    See Also
    --------
    mobgap.secondary_outcomes.GaitRegularityAndSymmetryTruncatedMean
        Regularity and symmetry with the nominal smoothing span and truncated edge windows.

    References
    ----------
    .. [1] R. Moe-Nilssen and J. L. Helbostad, "Estimation of gait cycle characteristics by trunk accelerometry,"
        J Biomech, vol. 37, no. 1, pp. 121-126, 2004.
    .. [2] D. Kobsar, C. Olson, R. Paranjape, T. Hadjistavropoulos, and J. M. Barden, "Evaluation of age-related
        differences in the stride-to-stride fluctuations, regularity and symmetry of gait using a waist-mounted
        tri-axial accelerometer," Gait Posture, vol. 39, no. 1, pp. 553-557, 2014.
    .. [3] L. M. A. van Gelder, L. Angelini, E. E. Buckley, and C. Mazza, "A proposal for a linear calculation of gait
        asymmetry," Symmetry, vol. 13, no. 9, p. 1560, 2021.
    """

    @timed_action_method
    @base_secondary_outcome_docfiller
    def calculate(self, data: pd.DataFrame, sampling_rate_hz: float, **_kwargs: Unpack[dict[str, Any]]) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(data_param)s
        %(sampling_rate_param)s

        %(calculate_return)s

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.signal_based_parameters_ = pd.Series(dtype=float)
        # return empty if not all accs are available
        required_acc_columns = ["acc_is", "acc_pa", "acc_ml"]
        if not all(col in data.columns for col in required_acc_columns):
            return self
        reg_sym = {}
        step_reg_is_loc = np.nan

        for axis in required_acc_columns:
            axis_results, step_reg_is_loc = self._compute_axis_metrics(
                data[axis].to_numpy(),
                axis=axis,
                sampling_rate_hz=sampling_rate_hz,
                step_reg_is_loc=step_reg_is_loc,
            )
            reg_sym.update(axis_results)

        self.signal_based_parameters_ = pd.Series(reg_sym, dtype=float)
        return self

    def _compute_axis_metrics(
        self,
        signal: np.ndarray,
        axis: str,
        sampling_rate_hz: float,
        step_reg_is_loc: float,
    ) -> tuple[dict[str, float], float]:
        """Compute regularity metrics for a single axis."""
        ax_direction = axis.split("_")[1]
        step_reg = stride_reg = asym_mn = sym_k = asym_g = np.nan
        distance = int(sampling_rate_hz / 4)
        try:
            n = len(signal)
            x = signal - signal.mean()
            norm = n - np.abs(np.arange(-n + 1, n))
            c = correlate(x, x, mode="full") / norm
            lags = np.arange(-n + 1, n)
            # normalise to zero-lag
            c = c / c[lags == 0][0]
            # non-negative lags
            c = c[lags >= 0]

            # in order to remove wrong detections of the irrelevant peaks the signal is smoothened.
            win_size = max(1, int((0.1 if axis == "acc_ml" else 0.2) * sampling_rate_hz))
            smoothed_c = self._smooth_autocovariance(c, win_size)

            # detect peaks
            if axis == "acc_ml":
                step_reg, stride_reg = self._process_ml_axis(c, smoothed_c, distance, step_reg_is_loc)
            else:
                step_reg, stride_reg, step_reg_is_loc = self._process_vt_ap_axis(
                    c, smoothed_c, distance, step_reg_is_loc
                )

            if not np.isnan(step_reg) and not np.isnan(stride_reg):
                asym_mn = abs(1 - step_reg / stride_reg)
                sym_k = abs(step_reg - stride_reg) / np.mean([step_reg, stride_reg])
                asym_g = (stride_reg + step_reg) / 2 if axis == "acc_ml" else (stride_reg - step_reg) / 2

        except (IndexError, ValueError):
            pass

        result = {
            f"step_regularity_{ax_direction}": step_reg,
            f"stride_regularity_{ax_direction}": stride_reg,
        }
        if ax_direction == "is":
            result.update(
                {
                    f"asymmetry_mn_{ax_direction}": asym_mn,
                    f"symmetry_k_{ax_direction}": sym_k,
                    f"asymmetry_g_{ax_direction}": asym_g,
                }
            )
        return result, step_reg_is_loc

    @staticmethod
    def _smooth_autocovariance(values: np.ndarray, span: int) -> np.ndarray:
        return _matlab_smooth_moving_ave(values, span)

    def _process_ml_axis(
        self, c: np.ndarray, smoothed_c: np.ndarray, distance: int, step_reg_is_loc: float
    ) -> tuple[float, float]:
        """Process ML axis."""
        # step regularity: negative peaks
        locs_neg, _ = find_peaks(-smoothed_c, distance=distance + 1)
        if not np.isnan(step_reg_is_loc):
            locs_neg = locs_neg[locs_neg >= step_reg_is_loc / 2]
        peaks_neg = -smoothed_c[locs_neg]
        mask = peaks_neg > 0
        locs_neg = locs_neg[mask]
        peaks_neg = peaks_neg[mask]
        _, corrected_peaks_neg = self._correct_peaks(-c, peaks_neg, locs_neg)
        step_reg = corrected_peaks_neg[0] if len(corrected_peaks_neg) > 0 else np.nan

        # stride regularity: positive peaks
        locs_pos, _ = find_peaks(smoothed_c, distance=distance + 1)
        if not np.isnan(step_reg_is_loc):
            locs_pos = locs_pos[locs_pos >= 1.5 * step_reg_is_loc]
        peaks_pos = smoothed_c[locs_pos]
        mask = peaks_pos > 0
        locs_pos = locs_pos[mask]
        peaks_pos = peaks_pos[mask]
        _, corrected_peaks_pos = self._correct_peaks(c, peaks_pos, locs_pos)
        stride_reg = corrected_peaks_pos[0] if len(corrected_peaks_pos) > 0 else np.nan

        return step_reg, stride_reg

    def _process_vt_ap_axis(
        self, c: np.ndarray, smoothed_c: np.ndarray, distance: int, step_reg_is_loc: float
    ) -> tuple[float, float, float]:
        """Process IS/PA axes."""
        locs, _ = find_peaks(smoothed_c, distance=distance + 1)
        peaks = smoothed_c[locs]
        mask = peaks >= 0
        locs = locs[mask]
        peaks = peaks[mask]
        locs, corrected_peaks = self._correct_peaks(c, peaks, locs)
        step_reg = corrected_peaks[0] if len(corrected_peaks) > 0 else np.nan
        stride_reg = corrected_peaks[1] if len(corrected_peaks) > 1 else np.nan
        if not np.isnan(step_reg) and np.isnan(step_reg_is_loc):
            step_reg_is_loc = locs[0] if len(locs) > 0 else np.nan
        return step_reg, stride_reg, step_reg_is_loc

    @staticmethod
    def _correct_peaks(data: np.ndarray, pks: np.ndarray, locs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Correct peaks found in a filtered signal to match original data."""
        if len(locs) < 2:
            return locs, pks

        median_diff = np.median(np.diff(locs))
        locale_win = int(np.ceil(0.2 * median_diff))

        # remove peaks too close to start/end
        mask = (locs > locale_win) & (locs < len(data) - locale_win)
        locs = locs[mask]

        # correct each peak to local maximum
        corrected_locs = []
        corrected_pks = []
        for loc in locs:
            start = max(loc - locale_win, 0)
            end = min(loc + locale_win // 2 + 1, len(data))
            window = data[start:end]
            max_idx = np.argmax(window)
            corrected_locs.append(start + max_idx)
            corrected_pks.append(window[max_idx])

        if len(corrected_locs) > 1:
            peaks = sorted(zip(corrected_locs, corrected_pks), key=lambda x: x[0])
            kept = []
            i = 0
            while i < len(peaks):
                j = i + 1
                while j < len(peaks) and peaks[j][0] - peaks[i][0] < locale_win:
                    j += 1
                cluster = peaks[i:j]
                best = max(cluster, key=lambda x: x[1])
                kept.append(best)
                i = j
            corrected_locs = np.array([p[0] for p in kept])
            corrected_pks = np.array([p[1] for p in kept])
        else:
            corrected_locs = np.array(corrected_locs)
            corrected_pks = np.array(corrected_pks)

        return corrected_locs, corrected_pks


def _matlab_smooth_moving_ave(y: np.ndarray, span: int) -> np.ndarray:
    """Replicate MATLAB's `smooth(y, span, 'moving')` function as closely as possible.

    1. It forces the span to be odd (e.g., span=20 becomes 19).
    2. It uses a non-standard "growing" window at the edges.
    """
    if span % 2 == 0:
        span -= 1
    n = len(y)
    half_span = (span - 1) // 2
    cum = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(y, out=cum[1:])
    i = np.arange(n)
    e = np.minimum(i + half_span + 1, n)
    mask_early = i < half_span
    e[mask_early] = 2 * i[mask_early] + 1
    s = np.maximum(0, i - half_span)
    mask_late = i >= (n - half_span)
    s[mask_late] = np.maximum(0, 2 * i[mask_late] - n + 1)
    sums = cum[e] - cum[s]
    window_sizes = e - s
    return sums / window_sizes


def _pd_smooth_moving_ave(y: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(y).rolling(span, center=True, min_periods=1).mean().to_numpy()
