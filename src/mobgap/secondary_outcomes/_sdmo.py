from typing import Any, Optional

import pandas as pd
import numpy as np
from scipy.signal import detrend, correlate, find_peaks
from typing_extensions import Self, Unpack

from numba import njit

from mobgap.secondary_outcomes.base import BaseSDMOCalculator, base_sdmo_docfiller
from mobgap.utils.dtypes import assert_is_sensor_data


@base_sdmo_docfiller
class SDMO(BaseSDMOCalculator):
    r"""Secondary digital mobility outcome calculations on IMU signal (ideally per walking bout).

    This "algorithm" calculates secondary outcomes for given signal window.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(secondary_outcomes)s

    """

    def __init__(
        self,
        *,
        replicate_matlab: bool,
    ) -> None:
        self.replicate_matlab = replicate_matlab
        self.smooth_moving_ave = _matlab_smooth_moving_ave if self.replicate_matlab else _pd_smooth_moving_ave

    @base_sdmo_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_para)s
        %(calculate_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        # expected the input data in body frame
        assert_is_sensor_data(self.data, frame="body")
        # collect all methods implementing SDMO calculation (add new ones to this list)
        # alternatively, inspect.getmembers can be used to get all methods (such as those starting with "_calculate")
        SDMO_functions = [self._calculate_rms, self._calculate_reg_sym]
        row = {"start": 0, "end": len(data)}
        for func in SDMO_functions:
            row.update(func(data).to_dict())
        self.secondary_outcomes = pd.DataFrame([row])
        return self

    def _calculate_rms(self, data: pd.DataFrame) -> pd.Series:
        """Compute acceleration, gyroscope, total acceleration signal root-mean-square (RMS), and ratio metrics.

        Ratio between RMS of axes i to RMSAccTotal (i = is, ml or pa)
        RMS ratio is based on the following article:
        A gait abnormality measure based on root mean square of trunk acceleration
        Masaki Sekine et al. Journal of NeuroEngineering and Rehabilitation 2013, 10:118
        http://www.jneuroengrehab.com/content/10/1/118
        """
        # first remove DC of acc signals
        data = data.copy()
        data.loc[:, data.columns.str.contains("acc")] = detrend(data.filter(like="acc").to_numpy(), axis=0)
        rms = (data.pow(2).mean() ** 0.5).add_prefix("RMS_")
        # total RMS
        rms_total_acc = ((rms[rms.index.str.contains("acc")]).pow(2).sum()) ** 0.5
        rms["RMSTotal_acc"] = rms_total_acc
        # ratio rms
        for key in [k for k in rms.keys() if k.startswith("RMS_acc") and k != "RMSTotal_acc"]:
            axis = key.replace("RMS_", "")
            rms[f"RMSRatio_{axis}"] = rms[key] / rms_total_acc if rms_total_acc != 0 else 0
        return rms

    def _calculate_reg_sym(self, data: pd.DataFrame) -> pd.Series:
        """Compute step/stride regularity and symmetry metrics from accelerations.

        Step/stride regularity and Assymetry_MN were developed according to:
        Estimation of gait cycle characteristics by trunk accelerometry. By Moe-Nilssen, Rolf, and Jorunn L. Helbostad.
        Journal of biomechanics 37, no. 1 (2004): 121-126. https://doi.org/10.1016/S0021-9290(03)00233-1

        Step Regularity - expression of the regularity of the acceleration signal
           between neighboring steps. Values range between [0-1] were closeness to 1
           means higher (better) regularity of the ait pattern. For ML only the
           value of the step regularity is provided as absolute value of the first negative peak.
        Stride Regularity - expression of the regularity of the acceleration signal
           between neighboring Strides. Values range between [0-1] were closeness to 1
           means higher (better) regularity of the ait pattern.
        Asymmetry_MN - step symmetry is defined as the ratio StepRegularity/StrideRegularity.
           closeness of the symmetry to 1 reflecs symmetry. Assymetry is defined
           as how close the ratio is to 1 and calculated as abs(1-symmetry).

        Please refer to section 2.5 in the article for additional discussion
        regarding the possible values of the 3 outcomes.

        Symmetry_K = 100*(absolute difference between step and stride regularity)/
                          (average of step and stride regularity).
        Symmetry_K was developed according to the following article:
        Gait Posture. 2014;39(1):553-7.
        doi: 10.1016/j.gaitpost.2013.09.008. Epub 2013 Sep 19.
        Value of zero means perfect symmetry and larger values depicting greater levels of step
        asymmetry in the accelerometer waveform.

        Asymmetry_G
        For V and AP:
        AS=(stride regularity-step regularity)/2
        For ML:
        AS=(stride regularity+step regularity)/2
        The suggested equation provides a linear scale, ranging between -1 and 1 and
        a value of 0 indicates perfect gait symmetry.
        Positive symmetry values relate to a gait pattern with a higher regularity of strides
        Negative values correspond to a gait pattern where the strides are more irregular than the steps.

        The article describing the Asymmetry_G measure can be found at:
        Van Gelder et al.A Proposal for a Linear Calculation of
        Gait Asymmetry. Symmetry 2021, 13,1560. https://doi.org/10.3390/sym13091560

        Units: all are unitless except for Symmetry_K which is provided as []
        """
        reg_sym = {}
        step_reg_is = np.nan
        distance = int(self.sampling_rate_hz / 4)
        for axis in ['acc_is', 'acc_pa', 'acc_ml']:
            signal = data[axis].to_numpy()
            step_reg = stride_reg = asym_mn = sym_k = asym_g = np.nan
            try:
                n = len(signal)
                x = signal - signal.mean()
                # normalized cross-covariance
                norm = n - np.abs(np.arange(-n + 1, n))
                c = correlate(x, x, mode='full') / norm
                lags = np.arange(-n + 1, n)
                # normalize c to zero-lag
                c = c / c[lags == 0][0]
                # non-negative lags
                c = c[lags >= 0]

                # in order to remove wrong detections of the irrelevant peaks the signal is smoothened.
                win_size = max(1, int((0.1 if axis == 'acc_ml' else 0.2) * self.sampling_rate_hz))
                smoothed_c = self.smooth_moving_ave(c, win_size)

                # detect peaks
                if axis == 'acc_ml':
                    # step regularity: negative peaks
                    locs, _ = find_peaks(-smoothed_c, distance=distance)
                    if not np.isnan(step_reg_is):
                        locs = locs[locs >= step_reg_is / 2]
                    peaks = -smoothed_c[locs]
                    locs = locs[peaks > 0]
                    peaks = peaks[peaks > 0]
                    _, peaks = _correct_peaks(-c, peaks, locs)
                    step_reg = peaks[0]

                    # stride regularity: positive peaks
                    locs, _ = find_peaks(smoothed_c, distance=distance)
                    if not np.isnan(step_reg_is):
                        locs = locs[locs >= 1.5 * step_reg_is]
                    peaks = smoothed_c[locs]
                    locs = locs[peaks > 0]
                    peaks = peaks[peaks > 0]
                    _, peaks = _correct_peaks(c, peaks, locs)
                    stride_reg = peaks[0]

                else:
                    # VT & AP axes
                    locs, _ = find_peaks(smoothed_c, distance=distance)
                    peaks = smoothed_c[locs]
                    locs = locs[peaks >= 0]
                    peaks = peaks[peaks >= 0]
                    locs, peaks = _correct_peaks(c, peaks, locs)
                    step_reg = peaks[0]
                    stride_reg = peaks[1]

                    if np.isnan(step_reg_is):
                        step_reg_is = locs[0]

                asym_mn = abs(1 - step_reg / stride_reg)
                sym_k = abs(step_reg - stride_reg) / np.mean([step_reg, stride_reg])
                asym_g = (stride_reg - step_reg) / 2

            except Exception:
                if axis == 'acc_is':
                    step_reg_is = np.nan

            ax_direction = axis.split("_")[1]
            reg_sym[f"StepRegularity_{ax_direction}"] = step_reg
            reg_sym[f"StrideRegularity_{ax_direction}"] = stride_reg
            if ax_direction == "is":
                reg_sym[f"Asymmetry_MN_{ax_direction}"] = asym_mn
                reg_sym[f"Symmetry_K_{ax_direction}"] = sym_k
                reg_sym[f"Asymmetry_G_{ax_direction}"] = asym_g

        return pd.Series(reg_sym)


    def _calculate_freq_amp_width_slope(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_jerk(self, data: pd.DataFrame) -> pd.Series:
        pass

    def _calculate_sd_range(self, data: pd.DataFrame) -> pd.Series:
        pass


def _matlab_smooth_moving_ave(y: np.ndarray, span: int) -> np.ndarray:
    """Replicate MATLAB's `smooth(y, span, 'moving')` function as closely as possible.
    1. It forces the span to be odd (e.g., span=20 becomes 19).
    2. It uses a non-standard "growing" window at the edges.
    """
    if span % 2 == 0:
        span -= 1
    y = np.asarray(y)
    n = len(y)
    yy = np.zeros(n)
    # half-width
    half_span = (span - 1) // 2
    for i in range(n):
        if i < half_span:
            # yy(k) = mean(y(1 : 2*k-1)) logic in MATLAB
            current_span = 2 * i + 1
            yy[i] = np.mean(y[0: current_span])
        elif i >= (n - half_span):
            j = (n - 1) - i
            current_span = 2 * j + 1
            yy[i] = np.mean(y[n - current_span: n])
        else:
            # standard
            yy[i] = np.mean(y[i - half_span: i + half_span + 1])
    return yy

def _pd_smooth_moving_ave(y: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(y).rolling(span, center=True, min_periods=1).mean().to_numpy()

def _correct_peaks(data: np.ndarray, pks: np.ndarray, locs: np.ndarray):
    """Correct peaks found in a filtered signal to match original data."""
    data = np.asarray(data)
    pks = np.asarray(pks)
    locs = np.asarray(locs)

    if len(locs) < 2:
        return pks, locs

    median_diff = np.median(np.diff(locs)) if len(locs) > 1 else 1
    locale_win = int(np.ceil(0.2 * median_diff))

    # remove peaks too close to start/end
    mask = (locs > locale_win) & (locs < len(data) - locale_win)
    locs = locs[mask]
    pks = data[locs]

    # correct each peak to local maximum
    corrected_locs = []
    corrected_pks = []
    for loc in locs:
        start = max(loc - locale_win, 0)
        end = min(loc + locale_win // 2, len(data))
        local_slice = data[start:end]
        Y = local_slice.max()
        I = local_slice.argmax()
        corrected_locs.append(start + I)
        corrected_pks.append(Y)

    corrected_locs = np.array(corrected_locs)
    corrected_pks = np.array(corrected_pks)

    # remove peaks that are too close
    if len(corrected_locs) > 1:
        close_peaks_idx = np.where(np.diff(corrected_locs) < locale_win)[0][::-1]
        for idx in close_peaks_idx:
            if corrected_pks[idx] > corrected_pks[idx + 1]:
                corrected_pks = np.delete(corrected_pks, idx + 1)
                corrected_locs = np.delete(corrected_locs, idx + 1)
            else:
                corrected_pks = np.delete(corrected_pks, idx)
                corrected_locs = np.delete(corrected_locs, idx)

    return corrected_locs, corrected_pks
