import warnings
from typing import Optional, Any
import numpy as np
import pandas as pd
from numba import njit
from scipy.fft import fft
from scipy.ndimage import minimum_filter1d
from scipy.signal import argrelextrema, correlate, detrend, find_peaks, medfilt, welch
from mobgap.signal_based.base import BaseSDMOCalculator, base_sdmo_docfiller
from typing_extensions import Self, Unpack


@base_sdmo_docfiller
class TurnSDMO(BaseSDMOCalculator):
    """Extract various features from the angular velocity signal around the Yaw axis during a turn.

    `turn_mean_ang_vel` is the amplitude of the mean angular velocity in the turn. Absolute value is used since the
    direction of the turn is not important.
    `turn_peak_ang_vel` is the absolute amplitude of the strongest extrema in the angular velocity of the turn
    `turn_smoothness` is the jerk of the angular velocity signal.
    `turn_dur_percentage_from_wb_dur` is the percentage of the total duration of the turning instances to the walking
    bout duration (the total duration of the given signal).
    """
    def __init__(self,):
        self.signal_based_parameters = pd.DataFrame([])


    def calculate(self, data: pd.DataFrame, sampling_rate_hz: float, turn_list: pd.DataFrame, **kwargs) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_return)s

        """
        if turn_list is None or turn_list.empty:
            return self
        turn_list = turn_list.copy()
        gyr = data["gyr_is"].to_numpy()
        means = []
        maxs = []
        # smoothness == jerk of yaw
        jerk_gyr = []
        for start, end, dur in turn_list[["start", "end", "duration_s"]].to_numpy():
            seg = gyr[int(start): int(end)]
            means.append(seg.mean())
            maxs.append(seg.max())
            jerk_gyr.append(np.sqrt(np.trapezoid(seg ** 2) / dur))

        turn_params = {
            "turn_mean_ang_vel": np.mean(means),
            "turn_peak_ang_vel": np.mean(maxs),
            "turn_smoothness": np.mean(jerk_gyr),
        }

        wb_dur = data.size / sampling_rate_hz
        turn_dur = turn_list["duration_s"].sum()
        turn_params["turn_dur_percentage_from_wb_dur"] = 100 * (turn_dur / wb_dur)
        self.signal_based_parameters = pd.DataFrame([turn_params])
        return self

@base_sdmo_docfiller
class StrideLevelSDMO(BaseSDMOCalculator):

    def __init__(
            self,
            stride_list_columns: list, # ["stride_length_m", "cadence_spm", "stride_duration_s"]
    ):
        self.stride_list_columns = stride_list_columns
        self.signal_based_parameters = pd.DataFrame([])

    def calculate(self, data:pd.DataFrame, stride_list: pd.DataFrame, **kwargs) -> Self:
        if stride_list is None:
            return self
        # in case the required columns are not available in the `stride_list`, then raise warning
        cols = self.stride_list_columns
        available_cols = [c for c in cols if c in stride_list.columns]
        if not available_cols:
            warnings.warn(
                f"Stride-level signal-based parameters are not calculated. None of {cols} is available in the"
                "stride list.",
                stacklevel=1,
            )
            return self
        cv = 100 * stride_list[available_cols].std() / stride_list[available_cols].mean()
        self.signal_based_parameters = cv.to_frame().T.add_prefix("cv_")
        return self

@base_sdmo_docfiller
class RMS(BaseSDMOCalculator):
    """Compute acceleration, gyroscope, total acceleration signal root-mean-square (RMS), and ratio metrics.

    Ratio between RMS of axes i to RMSAccTotal (i = is, ml or pa)
    RMS ratio is based on the following article:
    A gait abnormality measure based on root mean square of trunk acceleration
    Masaki Sekine et al. Journal of NeuroEngineering and Rehabilitation 2013, 10:118
    http://www.jneuroengrehab.com/content/10/1/118


    """
    def __init__(
            self,
    ):
        self.signal_based_parameters = pd.DataFrame([])

    def calculate(self, data: pd.DataFrame, **kwargs: Unpack[dict[str, Any]]) -> Self:
        if not any(data.columns.str.contains("acc")):
            return self
        # first remove DC of acc signals
        data = data.copy()
        data.loc[:, data.columns.str.contains("acc")] = detrend(data.filter(like="acc").to_numpy(), axis=0)
        rms = (data.pow(2).mean() ** 0.5).add_prefix("rms_")
        # total RMS
        rms_total_acc = ((rms[rms.index.str.contains("acc")]).pow(2).sum()) ** 0.5
        rms["rms_total_acc"] = rms_total_acc
        # ratio rms
        for key in rms.filter(like="rms_acc").index:
            rms[f"rms_ratio_{key.replace('rms_', '')}"] = rms[key] / rms_total_acc if rms_total_acc != 0 else 0
        self.signal_based_parameters = rms.to_frame().T
        return self

@base_sdmo_docfiller
class RegularitySymmetry(BaseSDMOCalculator):
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
    """
    def __init__(
            self,
    ):
        self.signal_based_parameters = pd.DataFrame([])

    def calculate(self, data: pd.DataFrame, sampling_rate_hz: float, replicate_matlab: bool, **kwargs) -> Self:  # noqa: PLR0915
        # return empty if not all accs are available
        required_acc_columns = ["acc_is", "acc_pa", "acc_ml"]
        if not all(col in data.columns for col in required_acc_columns):
            return self
        reg_sym = {}
        step_reg_is = np.nan
        distance = int(sampling_rate_hz / 4)
        for axis in required_acc_columns:
            signal = data[axis].to_numpy()
            step_reg = stride_reg = asym_mn = sym_k = asym_g = np.nan
            try:
                n = len(signal)
                x = signal - signal.mean()
                # normalized cross-covariance
                norm = n - np.abs(np.arange(-n + 1, n))
                c = correlate(x, x, mode="full") / norm
                lags = np.arange(-n + 1, n)
                # normalize c to zero-lag
                c = c / c[lags == 0][0]
                # non-negative lags
                c = c[lags >= 0]

                # in order to remove wrong detections of the irrelevant peaks the signal is smoothened.
                win_size = max(1, int((0.1 if axis == "acc_ml" else 0.2) * sampling_rate_hz))
                smooth_moving_func = _matlab_smooth_moving_ave if replicate_matlab else _pd_smooth_moving_ave
                smoothed_c = smooth_moving_func(c, win_size)

                # detect peaks
                if axis == "acc_ml":
                    # step regularity: negative peaks
                    locs, _ = find_peaks(-smoothed_c, distance=distance + 1)
                    if not np.isnan(step_reg_is):
                        locs = locs[locs >= step_reg_is / 2]
                    peaks = -smoothed_c[locs]
                    locs = locs[peaks > 0]
                    peaks = peaks[peaks > 0]
                    _, peaks = _correct_peaks(-c, peaks, locs)
                    step_reg = peaks[0]

                    # stride regularity: positive peaks
                    locs, _ = find_peaks(smoothed_c, distance=distance + 1)
                    if not np.isnan(step_reg_is):
                        locs = locs[locs >= 1.5 * step_reg_is]
                    peaks = smoothed_c[locs]
                    locs = locs[peaks > 0]
                    peaks = peaks[peaks > 0]
                    _, peaks = _correct_peaks(c, peaks, locs)
                    stride_reg = peaks[0]

                else:
                    # VT & AP axes
                    locs, _ = find_peaks(smoothed_c, distance=distance + 1)
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

            except Exception:  # noqa: BLE001
                if axis == "acc_is":
                    step_reg_is = np.nan

            ax_direction = axis.split("_")[1]
            reg_sym[f"step_regularity_{ax_direction}"] = step_reg
            reg_sym[f"stride_regularity_{ax_direction}"] = stride_reg
            if ax_direction == "is":
                reg_sym[f"asymmetry_mn_{ax_direction}"] = asym_mn
                reg_sym[f"symmetry_k_{ax_direction}"] = sym_k
                reg_sym[f"asymmetry_g_{ax_direction}"] = asym_g
        self.signal_based_parameters = pd.Series(reg_sym).to_frame().T
        return self

@base_sdmo_docfiller
class FrequencyAmplitudeWidthSlope(BaseSDMOCalculator):
    """Analyse the acceleration signal in the frequency domain.

    Calculate the max peak (center of the gait signal) amplitude and frequency, and the width (spread)
    of the gait frequencies around the main gait peak and the slope.

    More information:
    Toward Automated, At-Home Assessment of Mobility Among Patients With Parkinson Disease, Using a
    Body-Worn Accelerometer
    Aner Weiss et al.
    Neurorehabilitation and Neural Repair25(9) 810-818
    DOI: 10.1177/1545968311424869
    """
    def __init__(
            self,
            acc_columns: Optional[list[str]] = None, #["acc_is", "acc_ml", "acc_pa"]
    ):
        self.acc_columns = acc_columns
        self.signal_based_parameters = pd.DataFrame([])

    def calculate(self, data: pd.DataFrame, sampling_rate_hz:float, **kwargs) -> Self:
        acc = data.filter(like="acc")
        acc = (acc - acc.mean(axis=0)) / acc.std(axis=0).replace(0, 1)
        acc = acc[self.acc_columns].to_numpy()
        n = len(acc)
        fft_length = 2 ** (int(np.ceil(np.log2(n))) + 1)
        win_size = int(sampling_rate_hz * 2) if n >= 2 * sampling_rate_hz else n

        # welch PSD (should be close to the matlab's pwelch with the following params)
        def matlab_welch(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return welch(
                x, fs=sampling_rate_hz, window="hamming", nperseg=win_size, nfft=fft_length, detrend=False
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
        amp_is, freq_is, width_is, _slope_is = _extract_amp_freq_slope(psd_is, f, vap_freq_range)
        amp_ml, freq_ml, width_ml, _slope_ml = _extract_amp_freq_slope(psd_ml, f, ml_freq_range)
        amp_ap, freq_ap, width_ap, _slope_ap = _extract_amp_freq_slope(psd_ap, f, vap_freq_range)

        self.signal_based_parameters = pd.DataFrame(
            [{
                "amplitude_is": amp_is,
                "amplitude_ml": amp_ml,
                "amplitude_pa": amp_ap,
                "freq_is": freq_is,
                "freq_ml": freq_ml,
                "freq_pa": freq_ap,
                # the width and slope was commented out in the original implementation, but the sustain project
                # report lists width in the variability domain signal-based parameters, so return width default
                "width_is": width_is,
                "width_ml": width_ml,
                "width_pa": width_ap,
                # "slope_is": slope_is,
                # "slope_ml": slope_ml,
                # "slope_pa": slope_ap
            }]
        )
        return self


@base_sdmo_docfiller
class SampleEntropy(BaseSDMOCalculator):
    r"""Calculate Sample Entropy (SE) for accelerometer data.

    The Sample Entropy is defined as the negative natural average logarithm of the conditional probability
    that two sequences are similar for m points remain similar when the number of points is increased to m+1.
    Definition is taken from Torres (2013) [1]_. It can be used as a mean to describe the predictability of a
    signal.
    Useage for sway is given in Sofiane (2009) [2]_.

    .. [1] B.D.L.C. Torres, et al. "Entropy in the Analysis of Gait Complexity: A State of the Art". British Journal
    of Applied Science & Technology. 3(4) 1097-1105, 2013.
    .. [2] R. Sofiane, et al. "On the use of sample entropy to analyze human postural sway data". Medical
    Engineering & Physics. 31, 1023-1031, 2009.

    Parameters
    ----------
    dim
        ##
    r

    acc_columns

    num_samples_threshold

    Other Parameters
    ----------------
    (other_parameters)s

    Attributes
    ----------
    (signal_based_parameters)s

    """
    def __init__(
        self,
        *,
        dim: int = 2,
        r: float = 0.15,
        acc_columns: Optional[list[str]] = None, #["acc_is"]
        num_samples_threshold: int = 200,
    ) -> None:
        self.dim = dim
        self.r = r
        self.acc_columns = acc_columns
        self.num_samples_threshold = num_samples_threshold
        self.signal_based_parameters = pd.DataFrame([])

    def calculate(self, data: pd.DataFrame, **kwargs) -> Self:
        acc_columns = self.acc_columns
        if acc_columns is None:
            return self
        # dim: the sequence length that will be used for calculating the sample entropy. For gait dim=2 often used [1].
        # r: used for defining similarity between two sequences. Set to 0.15 as default in the original implementation
        dim = self.dim
        r = self.r
        # input data is downsampled by half
        accs = data[acc_columns].to_numpy()[::2]
        num_samples = accs.size

        # N=200 threshold is from [1]
        if num_samples <= self.num_samples_threshold:
            self.signal_based_parameters = pd.DataFrame([{f"sample_entropy_{col_name}": np.nan for col_name in acc_columns}])
            return self

        se_results = {}
        for acc, col_name in zip(accs.T, acc_columns):
            tol = r * np.std(acc)
            phi_m = np.mean(_phi(acc, dim, tol) / (num_samples - dim))
            phi_m1 = np.mean(_phi(acc, dim + 1, tol) / (num_samples - dim - 1))
            se_results[f"sample_entropy_{col_name}"] = -np.log(phi_m1 / phi_m)
        self.signal_based_parameters = pd.DataFrame([se_results])
        return self

@base_sdmo_docfiller
class HarmonicRatio(BaseSDMOCalculator):
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
    def __init__(
        self,
        *,
        acc_columns: Optional[list[str]] = None, #["acc_is", "acc_pa"]
    ) -> None:
        self.acc_columns = acc_columns
        self.signal_based_parameters = pd.DataFrame([])

    def calculate(self, data: pd.DataFrame, stride_list: pd.DataFrame, sampling_rate_hz:float, **kwargs) -> Self:  # noqa: C901, PLR0912, PLR0915
        # need the ic list
        ic_list = (stride_list["start"] - stride_list["start"].iloc[0]).to_numpy()
        acc_columns = self.acc_columns
        hr_results = {}
        if stride_list is None or len(ic_list) < 3:
            return self

        stride_pairs = list(zip(ic_list[::2], ic_list[2::2]))

        for col_name in acc_columns:
            acc = data[col_name].to_numpy()
            stride_harmonics = np.full((len(stride_pairs), 20), np.nan)
            is_ml = col_name == "acc_ml"
            gait_band = (0.25, 1.5) if is_ml else (0.5, 3.0)
            in_phase = np.arange(2, 19, 2) if is_ml else np.arange(3, 20, 2)
            out_phase = np.arange(1, 20, 2) if is_ml else np.arange(0, 19, 2)

            for stride_idx, (start, end) in enumerate(stride_pairs):
                current_end = end
                # flexing IC end point to eliminate high freq noise due to first and last sample amplitude mismatch
                start_points_in_data = np.where((acc[:-1] < acc[start]) & (acc[1:] >= acc[start]))[0]

                if start_points_in_data.size:
                    new_end = start_points_in_data[np.argmin(np.abs(start_points_in_data - current_end))]
                    stride_len = end - start + 1
                    if (current_end - new_end) <= 0.1 * stride_len:
                        current_end = new_end
                if start >= current_end:
                    # skip the stride
                    continue

                stride_data = acc[start : current_end + 1] - np.mean(acc[start : current_end + 1])
                # FFT
                n = len(stride_data)
                nfft = 2 ** (int(np.ceil(np.log2(n))) + 4)

                fft_vals = np.abs(fft(stride_data, nfft))[: nfft // 2 + 1]
                fft_freqs = np.linspace(0, sampling_rate_hz / 2, len(fft_vals))

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
                stride_time = n / sampling_rate_hz
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
                hr_results[f"harmonic_ratio_{col_name}"] = np.nan
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg = np.nanmean(stride_harmonics, axis=0)
                even_sum = np.nansum(avg[1::2])
                odd_sum = np.nansum(avg[0::2])
            hr_results[f"harmonic_ratio_{col_name}"] = even_sum / odd_sum if odd_sum else np.nan
        self.signal_based_parameters = pd.DataFrame([hr_results])
        return self


@base_sdmo_docfiller
class SDRange(BaseSDMOCalculator):
    def __init__(
        self,
    ) -> None:
        self.signal_based_parameters = pd.DataFrame([])


    def calculate(self, data: pd.DataFrame, **kwargs) -> Self:
        out = {}
        for c in data.columns:
            out[f"sd_{c}"] = data[c].std()
            if "acc" in c:  # range only for the acc columns
                out[f"range_{c}"] = data[c].max() - data[c].min()
        self.signal_based_parameters = pd.DataFrame([out])
        return self

@base_sdmo_docfiller
class Jerk(BaseSDMOCalculator):
    """Calculate jerk of acceleration and gyroscope signals in each principal direction, and log-normalised ratios.

    Jerk is defined as the third derivative of position with respect to time, so it is the second derivative
    of velocity and the first derivative of acceleration.
    In addition to the jerk of signals, log-normalized ratio of the jerk in AP vs jerk in the IS axis and
    jerk in ML vs jerk in the IS are calculated.

    Jerk ratio formula was taken from the following article:
    Age associated changes in head jerk while walking reveal altered dynamic stability in older people.
    Matthew A. et al., Exp Brain Res (2014) 232:51-60. DOI: 10.1007/s00221-013-3719-6

    Different methods of calculating jerk can be found in the following article:
    Sensitivity of smoothness measures to movement duration, amplitude, and arrests.
    Hogan N. et al., Journal of motor behavior (2009) 41,6. DOI:10.3200/35-09-004-RC
    """
    def __init__(
            self,
            acc_columns: Optional[list[str]] = None,  # ["acc_is", "acc_ml", "acc_pa"]
            gyr_columns: Optional[list[str]] = None,  # ["acc_is", "acc_ml", "acc_pa"]
    ) -> None:
        self.acc_columns = acc_columns
        self.gyr_columns = gyr_columns
        self.signal_based_parameters = pd.DataFrame([])

    def calculate(self, data: pd.DataFrame, sampling_rate_hz: float, **kwargs) -> Self:
        dt = 1 / sampling_rate_hz
        acc_dot = np.gradient(data[self.acc_columns].to_numpy(), dt, axis=0)
        integral_duration = dt * len(acc_dot)
        jerk_acc = np.sqrt(np.trapezoid(acc_dot**2, axis=0) / integral_duration)
        out = {
            **{f"jerk_{col}": jerk_acc[i] for i, col in enumerate(self.acc_columns)},
            # jerk acc ratio parameters are not reported in the sustain project report, so I commented them out
            # "JerkAccRatio_pa_is": 10 * np.log10(jerk_acc[2] / jerk_acc[0]),
            # "JerkAccRatio_ml_is": 10 * np.log10(jerk_acc[1] / jerk_acc[0]),
        }
        if set(self.gyr_columns).issubset(data.columns):
            gyr = data[self.gyr_columns].to_numpy().T
            jerk_gyr = np.sqrt(np.trapezoid(gyr**2, axis=1) / integral_duration)
            out.update(**{f"jerk_{col}": jerk_gyr[i] for i, col in enumerate(self.gyr_columns)})
        self.signal_based_parameters = pd.DataFrame([out])
        return self

@njit
def _phi(signal: np.ndarray, m: int, tol: float) -> np.ndarray:
    n = len(signal)
    n_dim = n - m + 1
    matches = np.zeros(n_dim, dtype=np.int32)
    for i in range(n_dim):
        count = 0
        for j in range(n_dim):
            if i == j:
                continue
            # Chebyshev distance (max absolute difference)
            d = 0.0
            for k in range(m):
                diff = abs(signal[i + k] - signal[j + k])
                if diff > d:
                    d = diff
                    if d > tol:  # early exit
                        break
            if d <= tol:
                count += 1
        matches[i] = count
    return matches


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


def _correct_peaks(data: np.ndarray, pks: np.ndarray, locs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correct peaks found in a filtered signal to match original data."""
    if len(locs) < 2:
        return pks, locs

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


def _extract_amp_freq_slope(
    psd: np.ndarray, freq: np.ndarray, freq_range: np.ndarray
) -> tuple[float, float, float, float]:
    """Extract amplitude, frequency, width and slope."""
    psd_sub = psd[freq_range]
    freq_sub = freq[freq_range]
    peaks, _ = find_peaks(psd_sub, distance=6)
    if len(peaks) == 0:
        return np.nan, np.nan, np.nan, np.nan

    peak = peaks[np.argmax(psd_sub[peaks])] if len(peaks) > 1 else peaks[0]

    amp = psd_sub[peak]
    freq_val = freq_sub[peak]
    half_amp = 0.5 * amp
    left_side = psd_sub[:peak]
    right_side = psd_sub[peak:]

    left_cross = np.where(left_side <= half_amp)[0]
    right_cross = np.where(right_side <= half_amp)[0]

    if len(left_cross) == 0 or len(right_cross) == 0:
        return amp, freq_val, np.nan, np.nan

    width_start = left_cross[-1]
    width_end = peak + right_cross[0]
    width = freq_sub[width_end] - freq_sub[width_start]
    smoothed = medfilt(psd_sub, kernel_size=5)
    minima = np.where(minimum_filter1d(smoothed, size=5) == smoothed)[0]
    pre_peak_min = minima[minima < peak]
    if len(pre_peak_min) == 0:
        return amp, freq_val, width, np.nan

    pre_peak = pre_peak_min[-1]
    rise_psd = psd_sub[pre_peak : peak + 1]
    rise_freq = freq_sub[pre_peak : peak + 1]
    range_val = amp - psd_sub[pre_peak]
    lower = psd_sub[pre_peak] + 0.25 * range_val
    upper = amp - 0.25 * range_val
    mask = (rise_psd >= lower) & (rise_psd <= upper)
    fit_psd = rise_psd[mask]
    fit_freq = rise_freq[mask]

    if len(fit_psd) < 2:
        return amp, freq_val, width, np.nan
    line_fit = np.polyfit(fit_freq, fit_psd, 1)
    slope = line_fit[0]

    return amp, freq_val, width, slope
