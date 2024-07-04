"""A set of universal constants."""

import scipy

#: Gravity in m/s^2
GRAV_MS2 = scipy.constants.g

#: Gyro cols in sensor frame
SF_GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]

#: Accel cols
SF_ACC_COLS = ["acc_x", "acc_y", "acc_z"]

#: Sensor cols
SF_SENSOR_COLS = [*SF_ACC_COLS, *SF_GYR_COLS]
