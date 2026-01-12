from typing import Any, Optional

import pandas as pd
import numpy as np
from scipy.signal import detrend, correlate, find_peaks, welch, medfilt, argrelextrema
from scipy.ndimage import minimum_filter1d
from scipy.fft import fft
from typing_extensions import Self, Unpack

from mobgap.signal_based.base import BaseSDMOCalculator, base_sdmo_docfiller
from mobgap.utils.dtypes import assert_is_sensor_data


@base_sdmo_docfiller
class SDMO(BaseSDMOCalculator):
    r"""Signal-based digital mobility outcome (SDMO) calculations on IMU signal (ideally per walking bout).

    This "algorithm" calculates SDMOs for given signal window.

    Other Parameters
    ----------------
    (other_parameters)s

    Attributes
    ----------
    (signal_based)s

    """
    replicate_matlab: bool

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
        ic_list: np.ndarray,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """(calculate_short)s.

        Parameters
        ----------
        (calculate_para)s
        (calculate_return)s
        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.ic_list = ic_list
        # expected the input data in body frame
        assert_is_sensor_data(self.data, frame="body")
        # collect all methods implementing SDMO calculation (add new ones to this list)
        # alternatively, inspect.getmembers can be used to get all methods (such as those starting with "_calculate")
        SDMO_functions = [self._calculate_rms, self._calculate_reg_sym, self._calculate_freq_amp_width_slope,
                          self._calculate_jerk, self._calculate_sd_range, self._calculate_harmonic_ratio,
                          self._calculate_sample_entropy]
        row = {"start": 0, "end": len(data)}
        for func in SDMO_functions:
            row.update(func(data).to_dict())
        self.signal_based_DMO = pd.DataFrame([row])
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
        """Analyse the acceleration signal in the frequency domain.

        Calculate the max peak (center of the gait signal) amplitude and frequency, and the width (spread)
        of the gait frequencies around the main gait peak and the slope.

        More information:
        Toward Automated, At-Home Assessment of Mobility Among Patients With Parkinson Disease, Using a
        Body-Worn Accelerometer
        Aner Weiss et al.
        Neurorehabilitation and Neural Repair25(9) 810–818
        DOI: 10.1177/1545968311424869
        """
        acc = data.filter(like="acc")
        acc = (acc - acc.mean(axis=0)) / acc.std(axis=0).replace(0, 1)
        acc = acc[["acc_is", "acc_ml", "acc_pa"]].to_numpy()
        n = len(acc)
        fft_length = 2 ** (int(np.ceil(np.log2(n))) + 1)
        if n >= 2 * self.sampling_rate_hz:
            win_size = int(self.sampling_rate_hz * 2)
        else:
            win_size = n
        # welch PSD (should be close to the matlab's pwelch with the following params)
        matlab_welch = lambda x: welch(
            x,
            fs=self.sampling_rate_hz,
            window='hamming',
            nperseg=win_size,
            nfft=fft_length,
            detrend=False
        )
        f, psd_is = matlab_welch(acc[:, 0])
        _, psd_ml = matlab_welch(acc[:, 1])
        _, psd_ap = matlab_welch(acc[:, 2])
        # frequency ranges
        fmin, fmax = 0.5, 3.0
        freq_delta = 0.1
        vap_freq_range = np.where((f >= fmin - freq_delta) & (f <= fmax + freq_delta))[0]
        ml_freq_range = np.where((f >= fmin / 2 - freq_delta) & (f <= fmax / 2 + freq_delta))[0]
        # extract amplitude, frequency, width and slope
        amp_is, freq_is, width_is, slope_is = _extract_amp_freq_slope(psd_is, f, vap_freq_range)
        amp_ml, freq_ml, width_ml, slope_ml = _extract_amp_freq_slope(psd_ml, f, ml_freq_range)
        amp_ap, freq_ap, width_ap, slope_ap = _extract_amp_freq_slope(psd_ap, f, vap_freq_range)

        return (
            pd.Series(
                {
                    "Amplitude_is": amp_is,
                    "Amplitude_ml": amp_ml,
                    "Amplitude_pa": amp_ap,
                    "Freq_is": freq_is,
                    "Freq_ml": freq_ml,
                    "Freq_pa": freq_ap,
                    # the width and slope was calculated but wasn't returned in the original implementation, so
                    # commented here, and in case they are required, they can be uncommented
                    # or maybe include a parameter (return_width, return_slope) to include them
                    # "Width_is": width_is,
                    # "Width_ml": width_ml,
                    # "Width_pa": width_ap,
                    # "Slope_is": slope_is,
                    # "Slope_ml": slope_ml,
                    # "Slope_pa": slope_ap
                }
            )
        )

    def _calculate_jerk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate jerk of acceleration and gyroscope signals in each principal direction, and log-normalised ratios.

        Jerk is defined as the third derivative of position with respect to time, so it is the second derivative
        of velocity and the first derivative of acceleration.
        In addition to the jerk of signals, log-normalized ratio of the jerk in AP vs jerk in the IS axis and
        jerk in ML vs jerk in the IS are calculated.

        Jerk ratio formula was taken from the following article:
        Age associated changes in head jerk while walking reveal altered dynamic stability in older people.
        Matthew A. et al., Exp Brain Res (2014) 232:51–60. DOI: 10.1007/s00221-013-3719-6

        Different methods of calculating jerk can be found in the following article:
        Sensitivity of smoothness measures to movement duration, amplitude, and arrests.
        Hogan N. et al., Journal of motor behavior (2009) 41,6. DOI:10.3200/35-09-004-RC
        """
        dt = 1 / self.sampling_rate_hz
        acc_columns = ["acc_is", "acc_ml", "acc_pa"]
        acc_dot = np.gradient(data[acc_columns].to_numpy(), dt, axis=0)
        integral_duration = dt * len(acc_dot)
        jerk_acc = np.sqrt(np.trapezoid(acc_dot ** 2, axis=0) / integral_duration)
        out = {
            **{f"Jerk_{col}": jerk_acc[i] for i, col in enumerate(acc_columns)},
            "JerkAccRatio_pa_is": 10 * np.log10(jerk_acc[2] / jerk_acc[0]),
            "JerkAccRatio_ml_is": 10 * np.log10(jerk_acc[1] / jerk_acc[0]),
        }
        gyr_columns = ["gyr_is", "gyr_ml", "gyr_pa"]
        if set(gyr_columns).issubset(data.columns):
            gyr = data[gyr_columns].to_numpy().T
            jerk_gyr = np.sqrt(np.trapezoid(gyr ** 2, axis=1) / integral_duration)
            out.update(**{f"Jerk_{col}": jerk_gyr[i] for i, col in enumerate(gyr_columns)})

        return pd.Series(out)

    def _calculate_sd_range(self, data: pd.DataFrame) -> pd.Series:
        out = {}
        for c in data.columns:
            out[f"SD_{c}"] = data[c].std()
            out[f"Range_{c}"] = data[c].max() - data[c].min()
        return pd.Series(out)


    def _calculate_harmonic_ratio(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the Harmonic Ratio (HR) for gait smoothness based on accelerometer data.

        HR is a measure of gait smoothness, based on the following article:
        Dynamic Stability in the Elderly: Identifying a Possible Measure
        H. John Yack et al., Journal of Gerontology: MEDICAL SCIENCES, 1993, Vol. 48, No. 5, M225-M230.

        The acceleration from the lower back contains repeatable patterns that contains information regarding
        the smoothness of the walking pattern. For acc_is and acc_ap accelerations a relatively larger ratio
        represents a smoother gait pattern. Their HR should be always greater than 1. The acc_ml acceleration has
        a monophasis pattern (one cycle in a stride vs 2 steps that are seen in other axes) which causes the first
        harmonic to be the dominant one. It has an HR smaller than 1.

        Stride time defines the period of the fundamental frequency component for calculating the smoothness
        of the signal. We calculate the first 20 harmonic coefficients using the finite fourier transform.
        Their amplitude is normalized using the fundamental frequency component amplitude. This process is
        performed for all the strides and then the harmonics are averaged across the strides. The ratio is
        calculated as the ratio of the even to odd harmonics.
        """
        acc_columns = ["acc_is", "acc_ml", "acc_pa"]
        hr_results = {}
        if len(self.ic_list) < 5:
            return pd.Series({f'HarmonicRatio_{k}': np.nan for k in acc_columns})

        stride_pairs = list(zip(self.ic_list[::2], self.ic_list[2::2]))

        for col_name in acc_columns:
            acc = data[col_name].to_numpy()
            stride_harmonics = np.full((len(stride_pairs), 20), np.nan)
            is_ml = (col_name == "acc_ml")
            gait_band = (0.25, 1.5) if is_ml else (0.5, 3.0)
            in_phase = np.arange(2, 19, 2) if is_ml else np.arange(3, 20, 2)
            out_phase = np.arange(1, 20, 2) if is_ml else np.arange(0, 19, 2)

            for stride_idx, (start, end) in enumerate(stride_pairs):
                # flexing IC end point to eliminate high freq noise due to first and last sample amplitude mismatch
                start_points_in_data = np.where(
                    (acc[:-1] < acc[start]) & (acc[1:] >= acc[start])
                )[0]

                if start_points_in_data.size:
                    new_end = start_points_in_data[np.argmin(np.abs(start_points_in_data - end))]
                    stride_len = end - start
                    if end != new_end:
                        # deviation 10% threshold
                        if (end - new_end) <= 0.1 * stride_len:
                            end = new_end
                if start >= end:
                    # skip the stride
                    continue

                stride_data = acc[start:end + 1] - np.mean(acc[start:end + 1])
                # FFT
                n = len(stride_data)
                nfft = 2 ** (int(np.ceil(np.log2(n))) + 4)

                fft_vals = np.abs(fft(stride_data, nfft))[:nfft // 2 + 1]
                fft_freqs = np.linspace(0, self.sampling_rate_hz / 2, len(fft_vals))

                max_idx = argrelextrema(fft_vals, np.greater)[0]
                if not max_idx.size:
                    continue

                max_freqs = fft_freqs[max_idx]
                max_amps = fft_vals[max_idx]
                band_mask = (max_freqs >= gait_band[0]) & (max_freqs <= gait_band[1])

                if not np.any(band_mask):
                    continue

                fundamental_idx = max_idx[band_mask][np.argmax(max_amps[band_mask])]
                fundamental_amp = fft_vals[fundamental_idx]
                fft_vals /= fundamental_amp
                stride_time = n / self.sampling_rate_hz
                f1 = (2 if is_ml else 1) / stride_time
                stride_harmonics[stride_idx, 0 if is_ml else 1] = 1.0
                harmonics = f1 * np.arange(1, 21)
                for h in in_phase:
                    f = harmonics[h]
                    mask = np.abs(fft_freqs[max_idx] - f) <= f1
                    if np.any(mask):
                        stride_harmonics[stride_idx, h] = fft_vals[max_idx[mask]].max()
                min_idx = argrelextrema(fft_vals, np.less)[0]
                min_idx = min_idx[fft_freqs[min_idx] >= 0.25]
                if min_idx.size:
                    min_freqs = fft_freqs[min_idx]
                    for h in out_phase:
                        f = harmonics[h]
                        mask = np.abs(min_freqs - f) <= f1
                        if np.any(mask):
                            stride_harmonics[stride_idx, h] = fft_vals[min_idx[mask]].min()
            if np.isnan(stride_harmonics).all():
                hr_results[f"HarmonicRatio_{col_name}"] = np.nan
            avg = np.nanmean(stride_harmonics, axis=0)
            even_sum = np.nansum(avg[1::2])
            odd_sum = np.nansum(avg[0::2])
            hr_results[f"HarmonicRatio_{col_name}"] = even_sum / odd_sum if odd_sum else np.nan

        return pd.Series(hr_results)


    def _calculate_sample_entropy(self, data: pd.DataFrame) -> pd.Series:
        r"""Calculate Sample Entropy (SE) for accelerometer data.

        The Sample Entropy is defined as the negative natural average logarithm of the conditional probability
        that two sequences are similar for m points remain similar when the number of points is increased to m+1.
        Definition is taken from Torres (2013) [1]_. It can be used as a mean to describe the predictability of a signal.
        Useage for sway is given in Sofiane (2009) [2]_.

        .. [1] B.D.L.C. Torres, et al. "Entropy in the Analysis of Gait Complexity: A State of the Art". British Journal
        of Applied Science & Technology. 3(4) 1097-1105, 2013.
        .. [2] R. Sofiane, et al. "On the use of sample entropy to analyze human postural sway data". Medical
        Engineering & Physics. 31, 1023–1031, 2009.

        """
        # dim: the sequence length that will be used for calculating the sample entropy. For gait dim=2 often used [1].
        # r: used for defining similarity between two sequences. Set to 0.15 as default in the original implementation
        dim = 2
        r = 0.15
        se_results = {}
        acc_columns = ["acc_is", "acc_ml", "acc_pa"]
        for col_name in acc_columns:
            # input data is downsampled by half in the original implementation
            acc = data[col_name].to_numpy()[::2]
            num_samples = acc.size
            # N=200 threshold is from [1]
            if num_samples <= 200:
                se_results[f"SampleEntropy_{col_name}"] = np.nan
                continue
            tol = r * np.std(acc)
            phi_m = np.mean(_phi(acc, dim, tol) / (num_samples - dim))
            phi_m1 = np.mean(_phi(acc, dim + 1, tol) / (num_samples - dim - 1))
            se_results[f"SampleEntropy_{col_name}"] = -np.log(phi_m1 / phi_m)
        return pd.Series(se_results)


def _phi(signal: np.ndarray, dim: float, tol:float) -> float:
    shape = (signal.size - dim + 1, dim)
    strides = (signal.strides[0], signal.strides[0])
    patterns = np.lib.stride_tricks.as_strided(signal, shape, strides)
    diff = np.abs(patterns[:, None, :] - patterns[None, :, :])
    dist = np.max(diff, axis=2)
    return np.sum(dist <= tol, axis=1) - 1


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

def _extract_amp_freq_slope(psd, freq, freq_range):
    """Extract amplitude, frequency, width and slope."""
    try:
        peaks, _ = find_peaks(psd[freq_range], distance=5)
        if len(peaks) == 0:
            return np.nan, np.nan, np.nan, np.nan
        if len(peaks) > 1:
            peak = peaks[np.argmax(psd[freq_range][peaks])]
        else:
            peak = peaks[0]

        amp = psd[freq_range][peak]
        freq_val = freq[freq_range][peak]

        left_candidates = np.where(psd[freq_range][:peak] <= 0.5 * amp)[0]
        right_candidates = np.where(psd[freq_range][peak:] <= 0.5 * amp)[0]

        if len(left_candidates) == 0 or len(right_candidates) == 0:
            return amp, freq_val, np.nan, np.nan

        width_start = left_candidates[-1]
        width_end = peak + right_candidates[0]
        width = freq[freq_range][width_end] - freq[freq_range][width_start]

        smoothed = medfilt(psd[freq_range], kernel_size=5)
        minima = np.where(minimum_filter1d(smoothed, size=5) == smoothed)[0]
        pre_peak_min = minima[minima < peak]
        if len(pre_peak_min) == 0:
            return amp, freq_val, width, np.nan
        pre_peak = pre_peak_min[-1]

        range_val = amp - psd[freq_range][pre_peak]
        slope_data = psd[freq_range][pre_peak:peak + 1]
        slope_freq = freq[freq_range][pre_peak:peak + 1]
        mask = (slope_data >= slope_data[0] + 0.25 * range_val) & (
                slope_data <= slope_data[-1] - 0.25 * range_val
        )
        slope_data = slope_data[mask]
        slope_freq = slope_freq[mask]
        if len(slope_data) < 2:
            return amp, freq_val, width, np.nan
        line_fit = np.polyfit(slope_freq, slope_data, 1)
        slope = line_fit[0]

        return amp, freq_val, width, slope
    except Exception:
        return np.nan, np.nan, np.nan, np.nan