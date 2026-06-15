"""
Reorientation algorithm for lower-back-worn IMU devices.

Corrects sensor axis orientation to the anatomical frame:
    IS  → vertical (infero-superior), pointing up
    ML  → mediolateral, pointing right
    PA  → posterior-anterior, pointing forward

Coordinate system: right-handed (IS up, ML right, PA forward).

Two correction modes:
    full - applies all three stages to every walking bout
    trust_gravity - skips ML/PA correction when gravity is already pointing
                    up in the vertical axis (is_up). Potential front/back
                    flips are ignored in this case.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy import signal
from tpcp import Algorithm, cf
from typing_extensions import Self, Unpack

from mobgap.data_transform import FirFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.re_orientation.base import BaseReorientationCorrector, base_reorientation_docfiller

GravityAxis = Literal["is", "ml"]
GravityDirection = Literal["up", "down"]
OrientationFamily = Literal["is_up", "is_down", "ml_up", "ml_down"]
GravityDetectionResult = tuple[Optional[GravityAxis], Optional[GravityDirection], Optional[OrientationFamily]]


# Results container
@dataclass
class ReorientationResult(BaseReorientationCorrector):
    """Stores detection output and the corrected data."""

    where_grav: Optional[GravityAxis]  # which device axis captured gravity
    where_grav_points: Optional[GravityDirection]  # direction of that axis
    family: Optional[OrientationFamily]  # orientation family
    phase: Optional[float]  # IS-PA phase value used for ML/PA correction; None if phase could not be computed
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
        full - applies ML/PA correction to every walking bout.
        trust_gravity - assumes mounting orientation is correct if gravity
        already points up along IS (``is_up``) and skips PA/ML sign correction.
        This intentionally ignores possible 180 deg front/back flips in this case.
    grav_threshold_ms2
        Minimum absolute mean acceleration in m/s² for an axis to be treated as
        capturing gravity.
    gait_frequency_band_filter
        The filter applied to ``acc_is`` and ``acc_pa`` before the cross-spectral
        phase is calculated.

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
    grav_threshold_ms2: float
    gait_frequency_band_filter: BaseFilter

    # Other Parameters
    data: pd.DataFrame
    sampling_rate_hz: float

    # Results
    result_: ReorientationResult

    def __init__(
        self,
        correction_mode: Literal["full", "trust_gravity"] = "trust_gravity",
        grav_threshold_ms2: float = 6.37,
        gait_frequency_band_filter: BaseFilter = cf(
            FirFilter(order=100, cutoff_freq_hz=(0.5, 2.5), filter_type="bandpass", zero_phase=True)
        ),
    ) -> None:
        self.correction_mode = correction_mode
        self.grav_threshold_ms2 = grav_threshold_ms2
        self.gait_frequency_band_filter = gait_frequency_band_filter

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
        where_grav, where_grav_points, family = _detect_gravity(data, self.grav_threshold_ms2)

        # Cannot correct if gravity not detected - return data unchanged
        if family is None:
            self.result_ = ReorientationResult(
                where_grav=None,
                where_grav_points=None,
                family=None,
                phase=None,
                correction_applied=False,
                correction_action="none",
                data_corrected=data.copy(),
            )
            return self

        # Stage 1: IS axis identity and direction correction
        if where_grav == "ml":
            corrected = _swap_is_ml(corrected)
            corrections.append("swapped IS-ML")
        if where_grav_points == "down":
            corrected = _flip_axes(corrected, ("is",))
            corrections.append("flipped IS")

        # trust_gravity: skip ML/PA correction when gravity is already correctly
        # aligned (is_up), as front/back flips are intentionally ignored
        if self.correction_mode == "trust_gravity" and family == "is_up":
            correction_action = " and ".join(corrections) if corrections else "none"
            self.result_ = ReorientationResult(
                where_grav=where_grav,
                where_grav_points=where_grav_points,
                family=family,
                phase=None,
                correction_applied=len(corrections) > 0,
                correction_action=correction_action,
                data_corrected=corrected,
            )
            return self

        # Stage 3: compute IS-PA phase on IS-corrected data
        phase = _cross_spec_pa_phase_power_weighted(corrected, sampling_rate_hz, self.gait_frequency_band_filter)

        # ML/PA correction based on family and phase sign
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

        return self

    @property
    def corrected_data_(self) -> pd.DataFrame:
        """The reoriented IMU data in the anatomical frame."""
        return self.result_.data_corrected


# Helper functions for each stage of the algorithm
def _detect_gravity(data: pd.DataFrame, grav_threshold_ms2: float) -> GravityDetectionResult:
    """
    Stage 1: identify which axis captures gravity.

    Stage 2: determine direction (up / down) and orientation family.

    Returns (where_grav, where_grav_points, family).
    family is None if no axis captures gravity.

    Possible families:
        ``is_up``   - gravity in IS axis, pointing up   (sensor correctly upright)
        ``is_down`` - gravity in IS axis, pointing down (sensor upside down)
        ``ml_up``   - gravity in ML axis, pointing up   (sensor rotated 90° sideways)
        ``ml_down`` - gravity in ML axis, pointing down (sensor rotated 90° sideways, inverted)
    """
    mean_is = data["acc_is"].mean()
    mean_ml = data["acc_ml"].mean()

    if abs(mean_is) >= grav_threshold_ms2:
        where_grav = "is"
        where_grav_points = "up" if mean_is > 0 else "down"
        family = f"{where_grav}_{where_grav_points}"

    elif abs(mean_ml) >= grav_threshold_ms2:
        where_grav = "ml"
        where_grav_points = "up" if mean_ml > 0 else "down"
        family = f"{where_grav}_{where_grav_points}"

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


def _cross_spec_pa_phase_power_weighted(
    data: pd.DataFrame, sampling_rate_hz: float, gait_frequency_band_filter: BaseFilter
) -> Optional[float]:
    """
    Compute power-weighted mean cross-spectral phase between acc_is and acc_pa.

    Computed across 0.5-2.5 Hz (gait stride frequency band).

    Signals are bandpass filtered before feature extraction.

    Positive → PA correctly oriented.
    Negative → PA reversed.
    Returns 0.0 if IS power in the stride band is zero (legitimate computed result).
    Returns None if phase cannot be computed (bout too short for filtering or
    spectral estimation, or no frequencies fall within the stride band).
    """
    # Apply bandpass filter before feature extraction
    try:
        filtered = (
            gait_frequency_band_filter.clone()
            .filter(data[["acc_is", "acc_pa"]], sampling_rate_hz=sampling_rate_hz)
            .transformed_data_
        )
    except ValueError as e:
        if "padlen" in str(e):
            return None
        raise e from None

    if len(filtered) < 4:
        return None

    acc_is_filt = filtered["acc_is"].to_numpy()
    acc_pa_filt = filtered["acc_pa"].to_numpy()

    nperseg = min(256, len(acc_is_filt) // 2)
    f, cxy = signal.csd(acc_is_filt, acc_pa_filt, fs=sampling_rate_hz, nperseg=nperseg)
    f, pxx_is = signal.welch(acc_is_filt, fs=sampling_rate_hz, nperseg=nperseg)

    stride_mask = (f >= 0.5) & (f <= 2.5)
    if not np.any(stride_mask):
        return None

    is_power = pxx_is[stride_mask]
    phase = np.angle(cxy[stride_mask])

    if is_power.sum() > 0:
        return float(np.average(phase, weights=is_power))

    return None


def _apply_ml_ap_correction(
    corrected: pd.DataFrame,
    family: OrientationFamily,
    phase: Optional[float],
) -> tuple[pd.DataFrame, Optional[str]]:
    """Apply ML/PA correction based on family and phase.

    If phase is None (could not be computed), no ML/PA correction is applied.
    """
    if phase is None:
        return corrected, "skipped ML/PA correction (phase unknown)"

    if family == "is_up":
        if phase < 0:
            return _flip_axes(corrected, ("ml", "pa")), "flipped ML and PA"

    elif family in {"is_down", "ml_up"}:
        if phase > 0:
            return _flip_axes(corrected, ("ml",)), "flipped ML"
        return _flip_axes(corrected, ("pa",)), "flipped PA"

    elif family == "ml_down" and phase < 0:
        return _flip_axes(corrected, ("ml", "pa")), "flipped ML and PA"

    return corrected, None
