"""
Reorientation algorithm for lower-back-worn IMU devices.

Corrects sensor axis orientation to the anatomical frame:
    IS  → vertical (infero-superior), pointing up
    ML  → mediolateral, pointing right
    AP  → anteroposterior, pointing forward

Coordinate system: left-handed (IS up, ML right, AP forward).

Two correction modes:
    full - applies all three stages to every walking bout
    trust_gravity - skips ML/AP correction when gravity is already pointing
                    up in the vertical axis (Family 1). Potential front/back
                    flips are ignored in this case.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy import signal
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap.data_transform import FirFilter
from mobgap.re_orientation.base import BaseReorientationCorrector, base_reorientation_docfiller

GRAVITY_THRESHOLD = 6.37  # m/s² - axis with |mean| >= captures gravity
_REORIENTATION_BANDPASS = FirFilter(order=100, cutoff_freq_hz=(0.5, 2.5), filter_type="bandpass", zero_phase=True)


# Results container
@dataclass
class ReorientationResult(BaseReorientationCorrector):
    """Stores detection output and the corrected data."""

    where_grav: Literal["is", "ml"]  # which device axis captured gravity
    where_grav_points: Literal["up", "down"]  # direction of that axis
    family: Literal[1, 2, 3, 4]  # orientation family
    phase: float  # IS-AP phase value used for ML/AP correction
    correction_applied: bool  # whether Stage 3 correction was applied
    correction_action: str  # description of correction applied, or 'none'
    data_corrected: pd.DataFrame = field(repr=False)  # corrected data


@base_reorientation_docfiller
class ReorientationMethodDM(Algorithm):
    """
    Detects and corrects IMU sensor orientation for lower-back-worn devices.

    Parameters
    ----------
    correction_mode : {'full', 'trust_gravity'}
        full - applies ML/AP correction to every walking bout.
        trust_gravity - assumes mounting orientation is correct if gravity
        already points up along IS and skips AP/ML sign correction. This
        intentionally ignores possible 180 deg front/back flips in this case.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(corrected_data_)s
    result_ : ReorientationResult
        Full detection and correction diagnostics including family, phase, correction flags.

    Examples
    --------
    >>> algo = ReorientationMethodDM(correction_mode="trust_gravity")
    >>> algo = algo.detect_correct(wb_data, sampling_rate_hz=100.0)
    >>> corrected = algo.result_.data_corrected
    """

    _action_methods = ("detect_correct",)

    # Parameters
    correction_mode: Literal["full", "trust_gravity"]

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # Results
    corrected_data_: pd.DataFrame
    result_: ReorientationResult

    def __init__(self, correction_mode: Literal["full", "trust_gravity"] = "trust_gravity") -> None:
        self.correction_mode = correction_mode

    @base_reorientation_docfiller
    def detect_correct(self, data: pd.DataFrame, *, sampling_rate_hz: float, **_: Unpack[dict[str, Any]]) -> Self:
        """%(detect_correct_short)s.

        Parameters
        ----------
        %(detect_correct_para)s

        %(detect_correct_return)s
        """
        # Validate correction_mode parameter
        if self.correction_mode not in ("full", "trust_gravity"):
            raise ValueError("correction_mode must be 'full' or 'trust_gravity'")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

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
            self.corrected_data_ = data.copy()
            return self

        # Stage 1: IS axis identity and direction correction
        if where_grav == "ml":
            corrected = _swap_is_ml(corrected)
            corrections.append("swapped IS-ML")
        if where_grav_points == "down":
            corrected = _flip_axes(corrected, ("is",))
            corrections.append("flipped IS")

        # trust_gravity: skip ML/AP correction for Family 1
        if self.correction_mode == "trust_gravity" and family == 1:
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
            self.corrected_data_ = corrected
            return self

        # Stage 3: compute IS-AP phase on IS-corrected data
        phase = _cross_spec_pa_phase_power_weighted(corrected, sampling_rate_hz)

        # ML/AP correction based on family and phase sign
        corrected, correction = _apply_ml_ap_correction(
            corrected,
            family,
            phase,
        )
        if correction is not None:
            corrections.append(correction)

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

        self.corrected_data_ = self.result_.data_corrected
        return self


# Helper functions for each stage of the algorithm
def _detect_gravity(data: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[int]]:
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


def _swap_is_ml(data: pd.DataFrame) -> pd.DataFrame:
    """Swap IS and ML axes (acc and gyr)."""
    out = data.copy()
    out[["acc_is", "acc_ml", "gyr_is", "gyr_ml"]] = data[["acc_ml", "acc_is", "gyr_ml", "gyr_is"]].to_numpy()
    return out


def _flip_axes(data: pd.DataFrame, axes: tuple[str, ...]) -> pd.DataFrame:
    """Negate one or more body axes (acc and gyr)."""
    out = data.copy()
    cols = [f"{sensor}_{axis}" for axis in axes for sensor in ("acc", "gyr")]
    out[cols] = -out[cols]
    return out


def _cross_spec_pa_phase_power_weighted(data: pd.DataFrame, sampling_rate_hz: float) -> float:
    """
    Compute power-weighted mean cross-spectral phase between acc_is and acc_pa.

    Computed across 0.5-3.0 Hz (gait stride frequency band).

    Signals are bandpass filtered (0.5-2.5 Hz) before feature extraction.

    Positive → AP correctly oriented.
    Negative → AP reversed.
    Returns 0.0 if bout is too short for spectral estimation or filtering.
    """
    # Check minimum length for filter (padlen = 3 * (order + 1) for filtfilt)
    min_length_for_filter = 3 * (_REORIENTATION_BANDPASS.order + 1)
    if len(data) < min_length_for_filter:
        return 0.0

    # Apply bandpass filter before feature extraction
    filtered = (
        _REORIENTATION_BANDPASS.clone()
        .filter(data[["acc_is", "acc_pa"]], sampling_rate_hz=sampling_rate_hz)
        .transformed_data_
    )
    acc_is_filt = filtered["acc_is"].to_numpy()
    acc_pa_filt = filtered["acc_pa"].to_numpy()

    nperseg = min(256, len(acc_is_filt) // 2)
    f, cxy = signal.csd(acc_is_filt, acc_pa_filt, fs=sampling_rate_hz, nperseg=nperseg)
    f, pxx_is = signal.welch(acc_is_filt, fs=sampling_rate_hz, nperseg=nperseg)

    stride_mask = (f >= 0.5) & (f <= 2.5)
    if not np.any(stride_mask):
        return 0.0

    is_power = pxx_is[stride_mask]
    phase = np.angle(cxy[stride_mask])

    if is_power.sum() > 0:
        return float(np.average(phase, weights=is_power))

    return 0.0


def _apply_ml_ap_correction(
    corrected: pd.DataFrame,
    family: int,
    phase: float,
) -> tuple[pd.DataFrame, Optional[str]]:
    """Apply ML/AP correction based on family and phase."""
    if family == 1:
        if phase < 0:
            return _flip_axes(corrected, ("ml", "pa")), "flipped ML and AP"

    elif family in {2, 3}:
        if phase > 0:
            return _flip_axes(corrected, ("ml",)), "flipped ML"
        return _flip_axes(corrected, ("pa",)), "flipped AP"

    elif family == 4 and phase < 0:
        return _flip_axes(corrected, ("ml", "pa")), "flipped ML and AP"

    return corrected, None
