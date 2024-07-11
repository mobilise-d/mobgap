from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import cf
from typing_extensions import Self, Unpack

from mobgap._gaitmap.trajectory_reconstruction.orientation_methods._madgwick import _madgwick_update_series
from mobgap.consts import INITIAL_MOBILISED_ORIENTATION, SF_ACC_COLS, SF_GYR_COLS
from mobgap.orientation_estimation.base import BaseOrientationEstimation
from mobgap.utils.conversions import to_sensor_frame
from mobgap.utils.dtypes import get_frame_definition

# Note: We completly extracted this from the vendored gaitmap package, as the logic was altered significantly. The base
# class was also extracted to the mobgap package. The underlying functions are still in the vendored modules.


class MadgwickAHRS(BaseOrientationEstimation):
    """The MadwickAHRS algorithm to estimate the orientation of an IMU.

    This method applies a simple gyro integration with an additional correction step that tries to align the estimated
    orientation of the z-axis with gravity direction estimated from the acceleration data.
    This implementation is based on the paper [1]_.
    An open source C-implementation of the algorithm can be found at [2]_.
    The original code is published under GNU-GPL.

    Parameters
    ----------
    beta
        This parameter controls how harsh the acceleration based correction is.
        A high value performs large corrections and a small value small and gradual correction.
        A high value should only be used if the sensor is moved slowly.
        A value of 0 is identical to just the Gyro Integration (see also
        :class:`gaitmap.trajectory_reconstruction.SimpleGyroIntegration` for a separate implementation).
    initial_orientation
        The initial orientation of the sensor that is assumed.
        It is critical that this value is close to the actual orientation.
        Otherwise, the estimated orientation will drift until the real orientation is found.
        In some cases, the algorithm will not be able to converge if the initial orientation is too far off and the
        orientation will slowly oscillate.
        If you pass a array, remember that the order of elements must be x, y, z, w.

        .. warning:: This orientation needs to be provided based on the global coordinate system, which is rotated
           relative to the sensor frame defined in mobgap.
           The defualt initial orientation is hence `mobgap.consts.INITIAL_MOBILISED_ORIENTATION`.
           If you want to provide a different initial orientation, you need to rotate it accordingly, depending on your
           definition.

    Attributes
    ----------
    orientation_
        The rotations as a *SingleSensorOrientationList*, including the initial orientation.
        This means the there are len(data) + 1 orientations.
    orientation_object_
        The orientations as a single scipy Rotation object
    rotated_data_
        The rotated data after applying the estimated orientation to the data.
        The first sample of the data remain unrotated (initial orientation).
        If the provided data was in the sensor frame, this will be in the normal global frame.
        If the provided data was in the body frame, this will be in the body-aligned global frame.

    Other Parameters
    ----------------
    data
        The data passed to the estimate method.
    sampling_rate_hz
        The sampling rate of this data

    Notes
    -----
    This class uses *Numba* as a just-in-time-compiler to achieve fast run times.
    In result, the first execution of the algorithm will take longer as the methods need to be compiled first.

    .. [1] Madgwick, S. O. H., Harrison, A. J. L., & Vaidyanathan, R. (2011).
           Estimation of IMU and MARG orientation using a gradient descent algorithm. IEEE International Conference on
           Rehabilitation Robotics, 1-7. https://doi.org/10.1109/ICORR.2011.5975346
    .. [2] http://x-io.co.uk/open-source-imu-and-ahrs-algorithms/

    Examples
    --------
    Your data must be a pd.DataFrame with columns defined by :obj:`~gaitmap.utils.consts.SF_COLS`.

    >>> import pandas as pd
    >>> from gaitmap.utils.consts import SF_COLS
    >>> data = pd.DataFrame(..., columns=SF_COLS)
    >>> sampling_rate_hz = 100
    >>> # Create an algorithm instance
    >>> mad = MadgwickAHRS(beta=0.2, initial_orientation=np.array([0, 0, 0, 1.0]))
    >>> # Apply the algorithm
    >>> mad = mad.estimate(data, sampling_rate_hz=sampling_rate_hz)
    >>> # Inspect the results
    >>> mad.orientation_
    <pd.Dataframe with resulting quaternions>
    >>> mad.orientation_object_
    <scipy.Rotation object>

    """

    initial_orientation: Union[np.ndarray, Rotation]
    beta: float

    def __init__(
        self,
        beta: float = 0.2,
        initial_orientation: Union[np.ndarray, Rotation] = cf(INITIAL_MOBILISED_ORIENTATION),
    ) -> None:
        self.initial_orientation = initial_orientation
        self.beta = beta

    def estimate(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """Estimate the orientation of the sensor.

        Parameters
        ----------
        data
            Continuous sensor data including gyro and acc values.
            The gyro data is expected to be in deg/s!
            The data can either be in the sensor or the body frame.
            Both will result in the same estimated sensor orientation, but the output of ``rotated_data_`` will be
            in the normal global frame if the data is in the sensor frame and in the body-aligned global frame if the
            data is in the body frame.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        frame = get_frame_definition(data, ["sensor", "body"])
        if frame == "body":
            data = to_sensor_frame(data)

        initial_orientation = self.initial_orientation

        if isinstance(initial_orientation, Rotation):
            initial_orientation = initial_orientation.as_quat()

        gyro_data = np.deg2rad(data[SF_GYR_COLS].to_numpy())
        acc_data = data[SF_ACC_COLS].to_numpy()
        madgwick_update_series = _madgwick_update_series
        rots = madgwick_update_series(
            gyro=gyro_data,
            acc=acc_data,
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            beta=self.beta,
        )

        self.orientation_object_ = Rotation.from_quat(rots)
        return self
