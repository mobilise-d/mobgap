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

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation
from tpcp import Algorithm, cf
from typing_extensions import Self, Unpack

from mobgap._gaitmap.utils.rotations import flip_dataset
from mobgap.data_transform import FirFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.re_orientation.base import BaseReorientationCorrector, base_reorientation_docfiller

GravityAxis = Literal["is", "ml"]
GravityDirection = Literal["up", "down"]
OrientationFamily = Literal["is_up", "is_down", "ml_up", "ml_down"]
ErrorHandling = Literal["raise", "warn", "ignore"]
UnresolvedReason = Literal["gravity", "pa_direction"]
GravityDetectionResult = tuple[Optional[GravityAxis], Optional[GravityDirection], Optional[OrientationFamily]]
_GRAVITY_ROTATIONS = {
    "is_up": Rotation.identity(),
    "is_down": Rotation.from_euler("z", 180, degrees=True),
    "ml_up": Rotation.from_euler("z", -90, degrees=True),
    "ml_down": Rotation.from_euler("z", 90, degrees=True),
}
_PA_DIRECTION_ROTATION = Rotation.from_euler("x", 180, degrees=True)
_IDENTITY_ROTATION = Rotation.identity()
_NO_GRAVITY_MESSAGE = (
    "No sensor axis with a clear gravity signal could be identified. No reorientation correction was applied. "
    "The data is returned uncorrected using the default Mobilise-D mounting assumption to transform it into the body "
    "frame."
)
_PA_DIRECTION_UNRESOLVED_MESSAGE = (
    "The direction of the PA axis could not be resolved from the phase estimate. The returned data only has the "
    "direction of gravity corrected. For the front-back flip, the PA axis is assumed to already point in the correct "
    "direction and is returned unmodified."
)


# Results container
@dataclass
class ReorientationResult(BaseReorientationCorrector):
    """Stores detection output, correction rotations, and corrected data.

    ``orientation_resolved`` is ``False`` when either gravity or the PA direction could not be detected. In non-raising
    modes, ``unresolved_reason`` identifies which part failed and ``data_corrected`` contains the documented fallback
    output.
    """

    where_grav: Optional[GravityAxis]  # which device axis captured gravity
    where_grav_points: Optional[GravityDirection]  # direction of that axis
    family: Optional[OrientationFamily]  # orientation family
    phase: Optional[float]  # IS-PA phase value used for ML/PA correction; None if phase could not be computed
    correction_applied: bool  # whether Stage 3 correction was applied
    correction_action: str  # description of correction applied, or 'none'
    orientation_resolved: bool  # whether all required orientation information was resolved
    unresolved_reason: Optional[UnresolvedReason]  # reason why orientation could not be fully resolved
    gravity_rotation: Rotation = field(repr=False)  # rotation applied to align gravity
    pa_direction_rotation: Rotation = field(repr=False)  # rotation applied to correct PA direction
    correction_rotation: Rotation = field(repr=False)  # combined correction rotation
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
    gravity_detection_error_type : {'raise', 'warn', 'ignore'}
        How to handle gait sequences where gravity can not be detected.
    pa_direction_detection_error_type : {'raise', 'warn', 'ignore'}
        How to handle gait sequences where the PA direction can not be detected.
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
        Full detection and correction diagnostics including family, phase, correction flags, unresolved detection
        reason, and the gravity/PA/combined correction rotations.

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
    gravity_detection_error_type: ErrorHandling
    pa_direction_detection_error_type: ErrorHandling
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
        gravity_detection_error_type: ErrorHandling = "warn",
        pa_direction_detection_error_type: ErrorHandling = "warn",
        gait_frequency_band_filter: BaseFilter = cf(
            FirFilter(order=100, cutoff_freq_hz=(0.5, 2.5), filter_type="bandpass", zero_phase=True)
        ),
    ) -> None:
        self.correction_mode = correction_mode
        self.grav_threshold_ms2 = grav_threshold_ms2
        self.gravity_detection_error_type = gravity_detection_error_type
        self.pa_direction_detection_error_type = pa_direction_detection_error_type
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
        _validate_error_handling(self.gravity_detection_error_type, "gravity_detection_error_type")
        _validate_error_handling(self.pa_direction_detection_error_type, "pa_direction_detection_error_type")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Stage 1+2: identify gravity axis, direction, and family
        where_grav, where_grav_points, family = _detect_gravity(data, self.grav_threshold_ms2)

        # Cannot correct if gravity not detected - return data unchanged
        if family is None:
            _handle_error(self.gravity_detection_error_type, _NO_GRAVITY_MESSAGE)
            self.result_ = ReorientationResult(
                where_grav=None,
                where_grav_points=None,
                family=None,
                phase=None,
                correction_applied=False,
                correction_action="none",
                orientation_resolved=False,
                unresolved_reason="gravity",
                gravity_rotation=_IDENTITY_ROTATION,
                pa_direction_rotation=_IDENTITY_ROTATION,
                correction_rotation=_IDENTITY_ROTATION,
                data_corrected=data.copy(),
            )
            return self

        gravity_rotation = _gravity_rotation(family)
        corrected = flip_dataset(data, gravity_rotation)
        corrections = _gravity_correction_actions(family)

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
                orientation_resolved=True,
                unresolved_reason=None,
                gravity_rotation=gravity_rotation,
                pa_direction_rotation=_IDENTITY_ROTATION,
                correction_rotation=gravity_rotation,
                data_corrected=corrected,
            )
            return self

        # Stage 3: compute IS-PA phase on IS-corrected data
        phase = _cross_spec_pa_phase_power_weighted(corrected, sampling_rate_hz, self.gait_frequency_band_filter)

        # ML/PA correction based on family and phase sign
        if phase is None:
            _handle_error(self.pa_direction_detection_error_type, _PA_DIRECTION_UNRESOLVED_MESSAGE)

        pa_direction_rotation = _pa_direction_rotation(phase)
        corrected = flip_dataset(corrected, pa_direction_rotation)
        correction = _pa_direction_correction_action(family, phase)
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
            orientation_resolved=phase is not None,
            unresolved_reason=None if phase is not None else "pa_direction",
            gravity_rotation=gravity_rotation,
            pa_direction_rotation=pa_direction_rotation,
            correction_rotation=pa_direction_rotation * gravity_rotation,
            data_corrected=corrected,
        )

        return self

    @property
    def corrected_data_(self) -> pd.DataFrame:
        """The reoriented IMU data in the anatomical frame."""
        return self.result_.data_corrected


def _validate_error_handling(error_type: str, parameter_name: str) -> None:
    if error_type not in ("raise", "warn", "ignore"):
        raise ValueError(f"{parameter_name} must be one of 'raise', 'warn', or 'ignore'.")


def _handle_error(error_type: ErrorHandling, message: str) -> None:
    if error_type == "raise":
        raise ValueError(message)
    if error_type == "warn":
        warnings.warn(message, UserWarning, stacklevel=2)


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


def _gravity_rotation(family: OrientationFamily) -> Rotation:
    return _GRAVITY_ROTATIONS[family]


def _gravity_correction_actions(family: OrientationFamily) -> list[str]:
    actions = {
        "is_up": [],
        "is_down": ["flipped IS"],
        "ml_up": ["swapped IS-ML"],
        "ml_down": ["swapped IS-ML", "flipped IS"],
    }
    return actions[family].copy()


def _pa_direction_rotation(phase: Optional[float]) -> Rotation:
    if phase is not None and phase < 0:
        return _PA_DIRECTION_ROTATION
    return Rotation.identity()


def _pa_direction_correction_action(family: OrientationFamily, phase: Optional[float]) -> Optional[str]:
    if phase is None:
        return "skipped ML/PA correction (phase unknown)"
    if phase < 0:
        if family in {"is_up", "ml_down"}:
            return "flipped ML and PA"
        return "flipped PA"
    if family in {"is_down", "ml_up"}:
        return "flipped ML"
    return None


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
