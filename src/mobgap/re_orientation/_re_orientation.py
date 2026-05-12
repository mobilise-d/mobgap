"""
Reorientation algorithm for lower-back-worn IMU devices.

Corrects sensor axis orientation to the anatomical frame:
    IS  → vertical (infero-superior), pointing up
    ML  → mediolateral, pointing right
    AP  → anteroposterior, pointing forward

Coordinate system: left-handed (IS up, ML right, AP forward).

Two methods:
    full - applies all three stages to every walking bout
    conservative - only applies ML/AP correction (Stage 3) when gravity
                   was already wrong (Family 2, 3, or 4). If gravity is
                   already pointing up in the vertical axis (Family 1),
                   ML and AP are left unchanged to avoid a ~6.67% risk
                   of wrongly flipping correctly oriented axes.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import signal
from tpcp import Algorithm
from typing_extensions import Self, Unpack

GRAVITY_THRESHOLD = 6.37  # m/s² - axis with |mean| >= this captures gravity
FS = 100  # sampling rate Hz


# Results container
@dataclass
class ReorientationResult:
    """Stores detection output and the corrected data."""

    where_grav: Literal["is", "ml"]  # which device axis captured gravity
    where_grav_points: Literal["up", "down"]  # direction of that axis
    family: Literal[1, 2, 3, 4]  # orientation family
    phase: float  # IS-AP phase value used for ML/AP correction
    correction_applied: bool  # whether Stage 3 correction was applied
    correction_action: str  # description of correction applied, or 'none'
    data_corrected: pd.DataFrame = field(repr=False)  # corrected data


class ReorientationMethodDM(Algorithm):
    """
    Detects and corrects IMU sensor orientation for lower-back-worn devices.

    Parameters
    ----------
    method : {'full', 'conservative'}
        full - applies ML/AP correction to every walking bout.
        conservative - skips ML/AP correction for Family 1 (gravity already
        pointing up in the vertical axis) to avoid wrongly flipping
        correctly oriented axes.

    Other Parameters
    ----------------
    data : pd.DataFrame
        The IMU data passed to the ``detect_correct`` method.

    Attributes
    ----------
    result_ : ReorientationResult
        The detection and correction result containing family, phase, correction flags, and corrected data.

    Examples
    --------
    >>> algo = ReorientationMethodDM(method="conservative")
    >>> algo = algo.detect_correct(wb_data)
    >>> corrected = algo.result_.data_corrected
    """

    _action_methods = ("detect_correct",)

    # Parameters
    method: Literal["full", "conservative"]

    # Other Parameters
    data: pd.DataFrame

    # Results
    result_: ReorientationResult

    def __init__(self, method: Literal["full", "conservative"] = "conservative"):
        self.method = method

    def detect_correct(self, data: pd.DataFrame, **_: Unpack[dict[str, Any]]) -> Self:
        """
        Detect orientation family and apply all corrections to a single walking bout.

        Parameters
        ----------
        data : pd.DataFrame
            Walking bout with columns: acc_is, acc_ml, acc_pa, gyr_is, gyr_ml, gyr_pa

        Returns
        -------
        self
            The class instance with ``result_`` attribute set.
        """
        # Validate method parameter
        if self.method not in ("full", "conservative"):
            raise ValueError("method must be 'full' or 'conservative'")

        self.data = data

        corrected = data.copy()
        corrections = []  # Track all corrections applied

        # Stage 1+2: identify gravity axis, direction, and family
        where_grav, where_grav_points, family = _detect_gravity(data)

        # Cannot correct if gravity not detected - return data unchanged
        if family is None:
            self.result_ = ReorientationResult(
                where_grav=None,
                where_grav_points=None,
                family=None,
                phase=0.0,
                correction_applied=False,
                correction_action="none",
                data_corrected=data.copy(),
            )
            return self

        # Stage 1: IS axis identity and direction correction
        if where_grav == "ml":
            corrected = _swap_is_ml(corrected, data)
            corrections.append("swapped IS-ML")
        if where_grav_points == "down":
            corrected = _flip_is(corrected)
            corrections.append("flipped IS")

        # Conservative: skip ML/AP correction for Family 1
        if self.method == "conservative" and family == 1:
            correction_action = " and ".join(corrections) if corrections else "none"
            self.result_ = ReorientationResult(
                where_grav=where_grav,
                where_grav_points=where_grav_points,
                family=family,
                phase=0.0,
                correction_applied=len(corrections) > 0,
                correction_action=correction_action,
                data_corrected=corrected,
            )
            return self

        # Stage 3: compute IS-AP phase on IS-corrected data
        phase = _cross_spec_pa_phase_power_weighted(corrected)

        # ML/AP correction based on family and phase sign
        if family == 1:
            if phase < 0:
                corrected = _flip_ml_and_ap(corrected)
                corrections.append("flipped ML and AP")

        elif family == 2 or family == 3:
            if phase > 0:
                corrected = _flip_ml(corrected)
                corrections.append("flipped ML")
            else:
                corrected = _flip_ap(corrected)
                corrections.append("flipped AP")

        elif family == 4:
            if phase < 0:
                corrected = _flip_ml_and_ap(corrected)
                corrections.append("flipped ML and AP")

        correction_action = " and ".join(corrections) if corrections else "none"

        self.result_ = ReorientationResult(
            where_grav=where_grav,
            where_grav_points=where_grav_points,
            family=family,
            phase=phase,
            correction_applied=len(corrections) > 0,
            correction_action=correction_action,
            data_corrected=corrected,
        )

        return self


# Helper functions for each stage of the algorithm
def _detect_gravity(data: pd.DataFrame):
    """
    Stage 1: identify which axis captures gravity.
    Stage 2: determine direction (up / down) and orientation family.

    Returns (where_grav, where_grav_points, family).
    family is None if no axis captures gravity.
    """
    mean_is = data["acc_is"].mean()
    mean_ml = data["acc_ml"].mean()

    if abs(mean_is) >= GRAVITY_THRESHOLD:
        where_grav = "is"
        where_grav_points = "up" if mean_is > 0 else "down"
        family = 1 if where_grav_points == "up" else 2

    elif abs(mean_ml) >= GRAVITY_THRESHOLD:
        where_grav = "ml"
        where_grav_points = "up" if mean_ml > 0 else "down"
        family = 3 if where_grav_points == "up" else 4

    else:
        where_grav = None
        where_grav_points = None
        family = None

    return where_grav, where_grav_points, family


def _swap_is_ml(corrected: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Swap IS and ML axes (acc and gyr)."""
    out = corrected.copy()
    out["acc_is"] = original["acc_ml"].copy()
    out["acc_ml"] = original["acc_is"].copy()
    out["gyr_is"] = original["gyr_ml"].copy()
    out["gyr_ml"] = original["gyr_is"].copy()
    return out


def _flip_is(data: pd.DataFrame) -> pd.DataFrame:
    """Negate IS axis (acc and gyr)."""
    out = data.copy()
    out["acc_is"] = -out["acc_is"]
    out["gyr_is"] = -out["gyr_is"]
    return out


def _flip_ml(data: pd.DataFrame) -> pd.DataFrame:
    """Negate ML axis (acc and gyr)."""
    out = data.copy()
    out["acc_ml"] = -out["acc_ml"]
    out["gyr_ml"] = -out["gyr_ml"]
    return out


def _flip_ap(data: pd.DataFrame) -> pd.DataFrame:
    """Negate AP axis (acc and gyr)."""
    out = data.copy()
    out["acc_pa"] = -out["acc_pa"]
    out["gyr_pa"] = -out["gyr_pa"]
    return out


def _flip_ml_and_ap(data: pd.DataFrame) -> pd.DataFrame:
    """Negate both ML and AP axes (acc and gyr)."""
    return _flip_ap(_flip_ml(data))


# Filter parameters
FILTER_LOWCUT = 0.5  # Hz
FILTER_HIGHCUT = 2.5  # Hz
FILTER_ORDER = 100


def _bandpass_filter(signal_data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int) -> np.ndarray:
    """
    Apply bandpass FIR filter to signal.

    Parameters
    ----------
    signal_data : np.ndarray
        Input signal
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    fs : int
        Sampling frequency (Hz)
    order : int
        Filter order

    Returns
    -------
    np.ndarray
        Filtered signal
    """
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design FIR bandpass filter
    fir_coeff = signal.firwin(order + 1, [low, high], pass_zero=False)

    # Apply filter
    filtered = signal.filtfilt(fir_coeff, 1.0, signal_data)

    return filtered


def _cross_spec_pa_phase_power_weighted(data: pd.DataFrame, fs: int = FS) -> float:
    """
    Power-weighted mean cross-spectral phase between acc_is and acc_pa
    across 0.5-3.0 Hz (gait stride frequency band).

    Signals are bandpass filtered (0.5-3.0 Hz) before feature extraction.

    Positive → AP correctly oriented.
    Negative → AP reversed.
    Returns 0.0 if bout is too short for spectral estimation.
    """
    acc_is = data["acc_is"].values
    acc_pa = data["acc_pa"].values

    if len(acc_is) < fs * 2:
        return 0.0

    try:
        # Apply bandpass filter before feature extraction
        acc_is_filt = _bandpass_filter(acc_is, FILTER_LOWCUT, FILTER_HIGHCUT, fs, FILTER_ORDER)
        acc_pa_filt = _bandpass_filter(acc_pa, FILTER_LOWCUT, FILTER_HIGHCUT, fs, FILTER_ORDER)

        nperseg = min(256, len(acc_is_filt) // 2)
        f, Cxy = signal.csd(acc_is_filt, acc_pa_filt, fs=fs, nperseg=nperseg)
        f, Pxx_is = signal.welch(acc_is_filt, fs=fs, nperseg=nperseg)
        stride_mask = (f >= 0.5) & (f <= 2.5)
        if not np.any(stride_mask):
            return 0.0
        is_power = Pxx_is[stride_mask]
        phase = np.angle(Cxy[stride_mask])
        if is_power.sum() > 0:
            return float(np.average(phase, weights=is_power))
    except Exception:
        pass

    return 0.0
