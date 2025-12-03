"""Helper functions for reorientation of IMU data using gaitmap."""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from gaitmap.utils.rotations import rotate_dataset

def acceleration_rot(IMU: pd.DataFrame, q: np.ndarray) -> np.ndarray:
   """Calculates the rotated acceleration based on the IMU data (accelerometry and gyroscope) and the rotation
   calculated from quaternions. This implementation employs the gaitmap library.
   For the use of Rotation.from_quat, we need to place the scalar part of the quaternion at the end of the array.

       Parameters
       ----------
       IMU
           The IMU data as an array of shape (n, 6).
       q
           The quaternions as an array of shape (n, 4).

       Returns
       -------
       The rotated acceleration as an array of shape (n, 3).

       """

   q_rot = np.hstack((q[:, 1:], q[:, :1]))
   rotations = Rotation.from_quat(q_rot)
   if len(IMU) == len(q):
      a = rotate_dataset(IMU, rotations)
      a = a.values
      a = a[:, 0:3]
   else:
      raise ValueError("The length of the IMU data and the quaternions must be the same.")
   return a
