import warnings
from typing import Any, Unpack

import numpy as np
import pandas as pd
from typing_extensions import Self

from gaitlink._docutils import make_filldoc
from gaitlink.cad.base import BaseCadenceCalculator
from gaitlink.data_transform import HampelFilter
from gaitlink.data_transform.base import BaseFilter
from gaitlink.utils.interpolation import interval_mean

ic2cad_docfiller = make_filldoc(
    {
        "ic2cad_short": """
    This uses a robust outlier removal approach to deal with missing initial contacts.
    The output cadence is reported as the average for each 1 second bin within the data.
    An incomplete second at the end is removed.

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
        "smoothing_filter": """
    step_time_smoothing
        The filter used to smooth the step time.
        This is used to remove outliers in the step time/cadence (e.g. when initial contacts are not detected).
        The filter is applied twice, once to the raw step time and a second time on the interpolated step time values
        per second.
        We recommend to use a Hampel filter for this.
    """,
    },
    doc_summary="Decorator to fill the explanation of the ic to cad per second interpolation.",
)


def _robust_ic_to_cad_per_sec(
    ics: pd.Series, sec_centers: np.ndarray, max_interp_gap_sec: int, smoothing_filter: BaseFilter
) -> pd.Series:
    """Calculate cadence per second from initial contacts."""
    ics = ics.to_numpy()
    if len(ics) <= 1:
        # We can not calculate cadence with only one initial contact
        return pd.Series(np.full(len(sec_centers), np.nan))
    step_time = np.diff(ics)
    # We repeat the last step time to get the same number of step times as initial contacts
    step_time = np.append(step_time, step_time[-1])
    # 1. Smoothing
    step_time_smooth = smoothing_filter.filter(step_time).transformed_data_
    # Average step time per second
    # TODO: Maybe shift the step time by half a step to get the step time at the center of the interval?
    sec_start_ends = np.vstack([sec_centers - 0.5, sec_centers + 0.5]).T
    step_time_per_sec = interval_mean(ics[:-1], step_time_smooth, sec_start_ends)
    # 2. Smoothing
    step_time_per_sec_smooth = smoothing_filter.filter(step_time_per_sec).transformed_data_
    # Interpolate missing values (only inside the recording and only if the gap is smaller than the specified maximum)
    step_time_per_sec_smooth = pd.Series(step_time_per_sec_smooth).interpolate(
        limit_area="inside", limit=max_interp_gap_sec
    )

    # Final cadence calculation
    return 1.0 / step_time_per_sec_smooth


@ic2cad_docfiller
class CadFromIc(BaseCadenceCalculator):
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
    %(smoothing_filter)s

    Notes
    -----
    %(ic2cad_notes)s

    """

    step_time_smoothing: BaseFilter
    max_interpolation_gap_s: int

    def __init__(
        self, *, step_time_smoothing: BaseFilter = HampelFilter(2, 3.0), max_interpolation_gap_s: int = 3
    ) -> None:
        self.max_interpolation_gap_s = max_interpolation_gap_s
        self.step_time_smoothing = step_time_smoothing

    def calculate(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.Series,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """Calculate cadence per second.

        Parameters
        ----------
        data
            The data represented as a dataframe.
        initial_contacts
            The initial contacts represented as a series.
        sampling_rate_hz
            The sampling rate of the IMU data in Hz.

        Returns
        -------
        Self
            The instance itself.
        """
        self.initial_contacts = initial_contacts
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        initial_contacts = self._get_ics(data, initial_contacts, sampling_rate_hz)
        n_secs = len(data) / sampling_rate_hz
        sec_centers = np.linspace(0, n_secs, len(data), endpoint=False) + 0.5 / sampling_rate_hz

        self.cadence_per_sec_ = _robust_ic_to_cad_per_sec(
            initial_contacts, sec_centers, self.max_interpolation_gap_s, self.step_time_smoothing.clone()
        )
        return self

    def _get_ics(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.Series,
        sampling_rate_hz: float,
    ) -> pd.Series:
        """Calculate initial contacts from the data."""
        if not initial_contacts.is_monotonic_increasing:
            raise ValueError("Initial contacts must be sorted in ascending order.")
        if initial_contacts.iloc[0] != 0 or initial_contacts.iloc[-1] != len(data):
            raise warnings.warn(
                "Usually we assume that gait sequences are cut to the first and last detected initial "
                "contact. "
                "This is not the case for the passed initial contacts and might lead to unexpected "
                "results in the cadence calculation. "
                "Specifically, you will get NaN values at the start and the end of the output.",
                stacklevel=3,
            )
        return initial_contacts


class CadFromIcDetector(CadFromIc):
    # TODO: correct typing
    ic_detector: None
    silence_ic_warning: bool

    ic_detector_: None

    def __init__(
        self,
        ic_detector,
        *,
        step_time_smoothing: BaseFilter = HampelFilter(2, 3.0),
        max_interpolation_gap_s: int = 3,
        silence_ic_warning: bool = False,
    ):
        self.ic_detector = ic_detector
        self.silence_ic_warning = silence_ic_warning
        super().__init__(step_time_smoothing=step_time_smoothing, max_interpolation_gap_s=max_interpolation_gap_s)

    @property
    def internal_initial_contacts_(self):
        return self.ic_detector_.initial_contacts_

    def _get_ics(
        self,
        data: pd.DataFrame,
        initial_contacts: pd.Series,
        sampling_rate_hz: float,
    ) -> pd.Series:
        if not self.silence_ic_warning:
            warnings.warn(
                "This method ignores the passed initial contacts and recalculates them from the data. "
                "This way you can use a different IC detector for the cadence calculation than for the IC detection. "
                "If you don't want this, you should use the CadFromIc class instead.\n\n"
                "This warning is just a information to make sure you are fully aware of this. "
                "If you want to silence this warning, you can pass ``silence_ic_warning=True`` during the "
                "initialization of this class.",
                stacklevel=3
            )

        self.ic_detector_ = self.ic_detector.clone().detect(data, sampling_rate_hz=sampling_rate_hz)
        new_initial_contacts = self.ic_detector_.initial_contacts_
        return new_initial_contacts
