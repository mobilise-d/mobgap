"""A set of universal constants and definitions."""

from typing import Final

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

#: Gravity in m/s^2
GRAV_MS2 = scipy.constants.g

#: Gyro cols in sensor frame
SF_GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]

#: Acc cols
SF_ACC_COLS = ["acc_x", "acc_y", "acc_z"]

#: Sensor cols
SF_SENSOR_COLS = [*SF_ACC_COLS, *SF_GYR_COLS]

#: Gyro cols in the body frame
BF_GYR_COLS = ["gyr_is", "gyr_ml", "gyr_pa"]

#: Acc cols in the body frame
BF_ACC_COLS = ["acc_is", "acc_ml", "acc_pa"]

#: Sensor cols in the body frame
BF_SENSOR_COLS = [*BF_ACC_COLS, *BF_GYR_COLS]

#: Acc cols in normal global frame
GF_ACC_COLS = ["acc_gx", "acc_gy", "acc_gz"]

#: Gyro cols in normal global frame
GF_GYR_COLS = ["gyr_gx", "gyr_gy", "gyr_gz"]

#: Sensor cols in normal global frame
GF_SENSOR_COLS = [*GF_ACC_COLS, *GF_GYR_COLS]

#: Acc cols in body frame aligned global frame
BGF_ACC_COLS = ["acc_gis", "acc_gml", "acc_gpa"]

#: Gyro cols in body frame aligned global frame
BGF_GYR_COLS = ["gyr_gis", "gyr_gml", "gyr_gpa"]

#: Sensor cols in body frame aligned global frame
BGF_SENSOR_COLS = [*BGF_ACC_COLS, *BGF_GYR_COLS]

COLS_PER_FRAME: Final = {
    "sensor": SF_SENSOR_COLS,
    "body": BF_SENSOR_COLS,
    "global": GF_SENSOR_COLS,
    "global_body": BGF_SENSOR_COLS,
}

#: The initial orientation of the Mobilise-D sensor frame in a "typical" global coordinate system.
INITIAL_MOBILISED_ORIENTATION = Rotation.from_matrix(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))
