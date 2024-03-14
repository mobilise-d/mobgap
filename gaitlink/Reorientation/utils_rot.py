"""Helper functions for reorientation of IMU data using gaitmap."""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from gaitmap.utils.rotations import rotate_dataset

def acceleration(IMU: pd.DataFrame, q: np.ndarray) -> np.ndarray:
   """Calculates the rotated acceleration based on the IMU data (accelerometry and gyroscope) and the rotation
   calculated from quaternions. This implementation employs the gaitmap library.

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

   rotations = Rotation.from_quat(q)
   if hasattr(rotations, '__iter__'):
      if IMU.shape[0] == len(rotations):
         a = rotate_dataset(IMU, rotations)
      a = a.values
      a = a[:, 0:3]
   else:
      a = rotate_dataset(IMU, rotations)
   return a
