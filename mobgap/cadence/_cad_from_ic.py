import warnings
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc
from mobgap.cadence.base import BaseCadCalculator, base_cad_docfiller
from mobgap.data_transform import HampelFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.initial_contacts import IcdHKLeeImproved, IcdShinImproved
from mobgap.initial_contacts.base import BaseIcDetector
from mobgap.utils.conversions import as_samples
from mobgap.utils.interpolation import robust_step_para_to_sec

ic2cad_docfiller = make_filldoc(
    {
        **base_cad_docfiller._dict,
        "ic2cad_short": """
    This uses a robust outlier removal approach to deal with missing initial contacts.
    The output cadence is reported as the average for each 1 second bin within the data.
    Note that an incomplete second at the end is included, to make sure that the entire data range is covered.

    Regions (i.e. second bouts) with no initial contacts are interpolated linearly based on the surrounding values, if
    the gap is smaller than the specified maximum interpolation gap.
    Regions without initial contacts that are larger than the specified maximum interpolation gap or at the very start
    or end of the recording are filled with NaNs.

    For more details see the Notes section.
    """,
        "ic2cad_notes": """
    The full process of calculating the cadence per second from initial contacts is as follows:

    1. We calculate the step time from the initial contacts.
       This results in one less step time than initial contacts.
       We replicate the last step time to get the same number of step times as initial contacts, to also provide a
       cadence value for the last initial contact for the last part of a gait sequence.
    2. We smooth the step time to remove outliers (using a Hampel filter by default).
    3. We calculate the step time per second by averaging the step time over the second.
       This is a little tricky, as some seconds might not contain any initial contacts.
       For all seconds that contain at least one initial contact, we calculate the average step time over the second.
       The step time for seconds without initial contacts is interpolated linearly based on the surrounding values.
       If the gap is larger than the specified maximum interpolation gap, the second is filled with NaNs.
       We don't fill "gaps" at the start and end of the recording, as we assume that gait sequences are cut anyway
       to start and end with a valid initial contact.
    4. We smooth the step time per second again to remove outliers on a per second level.
    5. We calculate the cadence per second by taking the inverse of the step time per second.

    In case there are less initial contacts than the window size of the smoothing filter, we return NaNs for all
    seconds.

    This approach deviates from the original Matlab implementations in a couple of ways:

    1. The original Matlab implementation calculates the cadence first (before any smoothing) or interpolation, while
       we perform all calculations on the step time and only calculate the cadence at the very end.
       The reasoning behind that is that because cadence is calculated by taking an inverse, it is much harder to reason
       about outliers in the cadence values.
       Because of this "projection" it is also much harder to reason about what the effect of linear interpolation
       on the cadence values is.
    2. The original Matlab implementation has no concept of "maximum interpolation gap".
       Values are interpolated by taking the average of the surrounding values, independent of the length of the gap.
       We decided to introduce a maximum interpolation gap to not mask "issues" in the ic-detection.
    3. When values are interpolated, the original Matlab implementation uses the average of the surrounding values.
       We decided to use linear interpolation instead, as this is more robust to outliers.
       These approaches are identical, if we only need to interpolate a single value, but for larger gaps, linear
       interpolation will start favoring the "closer" neighboring values, which we think is more appropriate.
    4. The original Matlab implementation extrapolates missing values at the start and end of the recording, by simply
       repeating the first/last value.
       This was done, to make sure that subsequent calculations don't need to deal with NaNs, but can easily mask issues
       in the ic-detection.
    """,
        "ic2cad_common_paras": """
    step_time_smoothing
        The filter used to smooth the step time.
        This is used to remove outliers in the step time/cadence (e.g. when initial contacts are not detected).
        The filter is applied twice, once to the raw step time and a second time on the interpolated step time values
        per second.
        We recommend to use a Hampel filter for this.
    max_interpolation_gap_s
        The maximum gap in seconds that is interpolated.
        If the gap is larger than this value, the second is filled with NaNs.
        We don't fill "gaps" at the start and end of the recording, as we assume that gait sequences are cut anyway
        to start and end with a valid initial contact.
    """,
        "smoothing_filter": """
    """,
    },
    doc_summary="Decorator to fill the explanation of the ic to cadence per second interpolation.",
)


def _robust_ic_to_cad_per_sec(
    ics: pd.Series, sec_centers: np.ndarray, max_interpolation_gap_s: int, smoothing_filter: BaseFilter
) -> np.ndarray:
    """Calculate cadence per second from initial contacts."""
    ics = ics.to_numpy()
    if len(ics) <= 1:
        # We can not calculate cadence with only one initial contact
        warnings.warn("Can not calculate cadence with only one or zero initial contacts.", stacklevel=3)
        return np.full(len(sec_centers), np.nan)
    step_time = np.diff(ics)
    # We repeat the last step time to get the same number of step times as initial contacts
    step_time = np.append(step_time, step_time[-1])
    step_time_per_sec_smooth = robust_step_para_to_sec(
        ics,
        step_time,
        sec_centers,
        max_interpolation_gap_s,
        smoothing_filter,
    )

    # Final cadence calculation in 1/min
    return 1.0 / step_time_per_sec_smooth * 60


@ic2cad_docfiller
class CadFromIc(BaseCadCalculator):
    """Calculate cadence per second directly from initial contacts.

    This method will directly take the initial contacts provided in to the ``calculate`` method and calculate the
    cadence per second from them.
    We further assume that the initial contacts are sorted and that the first initial contact is the start of the
    passed data and the last initial contact is the end of a passed data.

    .. note :: This method does not recalculate the initial contacts, but uses the passed initial contacts directly.
       Hence, it assumes that you want to use the same IC detector for the cadence calculation as for the IC detection.
       If you want to use a different IC detector for the cadence calculation, you should use the
       :class:`CadFromIcDetector`, which will internally use the passed IC detector to calculate the initial contacts
       again just for the cadence calculation.

    %(ic2cad_short)s

    Parameters
    ----------
    %(ic2cad_common_paras)s

    Attributes
    ----------
    %(cadence_per_sec_)s

    Other Parameters
    ----------------
    %(other_parameters)s

    Notes
    -----
    %(ic2cad_notes)s

    """

    step_time_smoothing: BaseFilter
    max_interpolation_gap_s: int

    def __init__(
        self, *, step_time_smoothing: BaseFilter = cf(HampelFilter(2, 3.0)), max_interpolation_gap_s: int = 3
    ) -> None:
        self.max_interpolation_gap_s = max_interpolation_gap_s
        self.step_time_smoothing = step_time_smoothing

    @ic2cad_docfiller
    def calculate(
        self,
        data: pd.DataFrame,
        *,
        initial_contacts: pd.DataFrame,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """%(calculate_short)s.

        Parameters
        ----------
        %(calculate_para)s

        %(calculate_return)s
        """
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        initial_contacts = self._get_ics(data, initial_contacts, sampling_rate_hz)["ic"]
        if not initial_contacts.is_monotonic_increasing:
            raise ValueError("Initial contacts must be sorted in ascending order.")
        initial_contacts_in_seconds = initial_contacts / sampling_rate_hz
        n_secs = len(data) / sampling_rate_hz
        sec_centers = np.arange(0, n_secs) + 0.5
        self.cadence_per_sec_ = pd.DataFrame(
            {
                "cadence_spm": _robust_ic_to_cad_per_sec(
                    initial_contacts_in_seconds,
                    sec_centers,
                    self.max_interpolation_gap_s,
                    self.step_time_smoothing.clone(),
                )
            },
            index=as_samples(sec_centers, sampling_rate_hz),
        ).rename_axis(index="sec_center_samples")
        return self

    def _get_ics(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.DataFrame,
        sampling_rate_hz: float,  # noqa: ARG002
    ) -> pd.DataFrame:
        """Calculate initial contacts from the data."""
        ic_series = initial_contacts["ic"]
        if len(ic_series) == 0:
            warnings.warn("No initial contacts provided. Cad will be NaN", stacklevel=2)
        if len(ic_series) > 0 and (ic_series.iloc[0] != 0 or ic_series.iloc[-1] != len(data) - 1):
            warnings.warn(
                "Usually we assume that gait sequences are cut to the first and last detected initial "
                "contact. "
                "This is not the case for the passed initial contacts and might lead to unexpected "
                "results in the cadence calculation. "
                "Specifically, you will get NaN values at the start and the end of the output.",
                stacklevel=2,
            )
        return initial_contacts


@ic2cad_docfiller
class CadFromIcDetector(CadFromIc):
    """Calculate cadence per second by detecting initial contacts using a provided IC detector.

    .. warning :: This method ignores the passed initial contacts and recalculates them from the data using the passed
       IC detector.
       If you want to use the ICs passed to the ``calculate`` method, you should use the :class:`CadFromIc` class.

    This method will first calculate the initial contacts using the passed IC detector and then calculate the cadence
    per second from them.

    %(ic2cad_short)s

    Parameters
    ----------
    ic_detector
        The IC detector used to detect the initial contacts.
    %(ic2cad_common_paras)s
    silence_ic_warning
        By default the method warns you, that it ignores the passed initial contacts and recalculates them from the
        data using the passed IC detector.
        We do that, as it is likely that users forget about this and might be surprised by the results.
        If you are aware of this and want to silence this warning, you can pass ``silence_ic_warning=True``.

    Attributes
    ----------
    %(cadence_per_sec_)s
    ic_detector_
        The IC detector used to detect the initial contacts with the results attached.
        This is only available after the ``detect`` method was called.
    internal_ic_list_
        The initial contacts detected by the ``ic_detector_``.
        This is equivalent to ``ic_detector_.initial_contacts_``.

    Other Parameters
    ----------------
    %(other_parameters)s

    Notes
    -----
    %(ic2cad_notes)s

    """

    ic_detector: BaseIcDetector
    silence_ic_warning: bool

    ic_detector_: BaseIcDetector

    class PredefinedParameters:
        """Predefined parameters for the :class:`CadFromIc` class.

        These are used to create instances of the :class:`CadFromIc` class with the recommended parameter for regular
        and impaired walking.
        """

        _base_paras: Final = MappingProxyType(
            {
                "step_time_smoothing": HampelFilter(2, 3.0),
                "max_interpolation_gap_s": 3,
                "silence_ic_warning": False,
            }
        )

        regular_walking: Final = MappingProxyType({"ic_detector": IcdShinImproved(), **_base_paras})
        impaired_walking: Final = MappingProxyType({"ic_detector": IcdHKLeeImproved(), **_base_paras})

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.regular_walking.items()})
    def __init__(
        self,
        ic_detector: BaseIcDetector,
        *,
        step_time_smoothing: BaseFilter,
        max_interpolation_gap_s: int,
        silence_ic_warning: bool,
    ) -> None:
        self.ic_detector = ic_detector
        self.silence_ic_warning = silence_ic_warning
        super().__init__(step_time_smoothing=step_time_smoothing, max_interpolation_gap_s=max_interpolation_gap_s)

    @property
    def internal_ic_list_(self) -> pd.DataFrame:
        return self.ic_detector_.ic_list_

    def _get_ics(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.DataFrame,  # noqa: ARG002
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
        if not self.silence_ic_warning:
            warnings.warn(
                "This method ignores the passed initial contacts and recalculates them from the data. "
                "This way you can use a different IC detector for the cadence calculation than for the IC detection. "
                "If you don't want this, you should use the `CadFromIc` class instead.\n\n"
                "This warning is just a information to make sure you are fully aware of this. "
                "If you want to silence this warning, you can pass ``silence_ic_warning=True`` during the "
                "initialization of this class.",
                stacklevel=3,
            )

        self.ic_detector_ = self.ic_detector.clone().detect(data, sampling_rate_hz=sampling_rate_hz)
        if len(self.internal_ic_list_) == 0:
            warnings.warn(
                "The interal IC detector did not detect any initial contacts provided. Cad will be NaN", stacklevel=2
            )
        return self.internal_ic_list_
