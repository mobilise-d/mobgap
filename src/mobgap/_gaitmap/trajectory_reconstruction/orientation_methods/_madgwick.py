"""Implementation of the MadgwickAHRS."""

import numpy as np
from numba import njit

from mobgap._gaitmap.utils.fast_quaternion_math import rate_of_change_from_gyro


@njit()
def _madgwick_update(gyro, acc, initial_orientation, sampling_rate_hz, beta):
    q = np.copy(initial_orientation)
    qdot = rate_of_change_from_gyro(gyro, q)

    # Note that we change the order of q components here as we use a different quaternion definition.
    q1, q2, q3, q0 = q

    if beta > 0.0 and not np.all(acc == 0.0):
        acc = acc / np.sqrt(np.sum(acc**2))
        ax, ay, az = acc

        # Auxiliary variables to avoid repeated arithmetic
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        # Gradient decent algorithm corrective step
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
        s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay

        # Switch the component order back
        s = np.array([s1, s2, s3, s0])
        mag_s = np.sqrt(np.sum(s**2))
        if mag_s != 0.0:
            s /= np.sqrt(np.sum(s**2))

        # Apply feedback step
        qdot -= beta * s

    # Integrate rate of change of quaternion to yield quaternion
    q = q + qdot / sampling_rate_hz
    q /= np.sqrt(np.sum(q**2))

    return q


@njit(cache=True)
def _madgwick_update_series(gyro, acc, initial_orientation, sampling_rate_hz, beta):
    out = np.empty((len(gyro) + 1, 4))
    out[0] = initial_orientation
    for i, (gyro_val, acc_val) in enumerate(zip(gyro, acc)):
        initial_orientation = _madgwick_update(
            gyro=gyro_val,
            acc=acc_val,
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            beta=beta,
        )
        out[i + 1] = initial_orientation

    return out
