"""Helper functions for reorientation of IMU data."""

import numpy as np
import pandas as pd


def acceleration(
        IMU: pd.DataFrame,
        q: np.ndarray
) -> pd.DataFrame:
    """Calculates the rotated acceleration based on the IMU data and the quaternions.

    Parameters
    ----------
    IMU
        The IMU data as an array of shape (n, 3).
    q
        The quaternions as an array of shape (n, 4).

    Returns
    -------
    The rotated acceleration as a pd.DataFrame.

    Raises
    ------
    ValueError
        If the number of columns of IMU is not 3.
        If the number of columns of q is not 4.
        If the number of rows of IMU and q are not the same.

    """
    if IMU.shape[1] != 3:
        raise ValueError(f"IMU must have 3 columns, but has {IMU.shape[1]}")

    if q.shape[1] != 4:
        raise ValueError(f"q must have 4 columns, but has {q.shape[1]}")

    if IMU.shape[0] != q.shape[0]:
        raise ValueError(
            f"IMU and q must have the same number of rows, but have {IMU.shape[0]} and {q.shape[0]} respectively")

    a = np.zeros((len(q), 3))
    for i in range(len(q)):
        a[i, :] = quaterot(q[i, :], IMU.iloc[i, 0:3])
    return pd.DataFrame(a, columns=['acc_x', 'acc_y', 'acc_z'])


def quaterot(
        q: np.ndarray,
        acc: np.ndarray
) -> np.ndarray:
    """Rotates the acceleration using the quaternion.

    Parameters
    ----------
    q
        The quaternion as an array of shape (4,).
    acc
        The acceleration as an array of shape (3,).

    Returns
    -------
    The rotated acceleration as an array of shape (3,).

    Raises
    ------
    ValueError
        If the number of columns of q is not 4.
        If the number of columns of acc is not 3.
        If the number of rows of acc and q are not the same.

    """

    if q.shape[1] != 4:
        raise ValueError(f"q must have 4 columns, but has {q.shape[1]}")

    if acc.shape[1] != 3:
        raise ValueError(f"acc must have 3 columns, but has {acc.shape[1]}")

    if q.shape[0] != acc.shape[0]:
        raise ValueError(
            f"acc and q must have the same number of rows, but have {acc.shape[0]} and {q.shape[0]} respectively")

    qacc = quatmultiply(q, quatmultiply(np.array([0, *acc]), conj(q)))
    qacc = qacc[1:4].reshape(-1, 1).flatten()
    return qacc


def conj(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion. Python implementation of "quatconj" in MATLAB.

    Parameters:
    ----------
    q (np.ndarray)
        Quaternion as an array of shape (4,).

    Returns
    -------
    np.ndarray: Conjugate of the input quaternion.

    Raises
    ------
    ValueError
        If the number of columns of q is not 4.

    """

    if q.shape[1] != 4:
        raise ValueError(f"q must have 4 columns, but has {q.shape[1]}")

    conj = q.copy()
    conj[1:][conj[1:] != 0] *= -1
    return conj


def quatmultiply(r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Computes the product of two quaternions. Python implementation of "quatmultiply" in MATLAB.

    Parameters:
    ----------
    r (np.ndarray)
        Quaternion as an array of shape (4,).

    q (np.ndarray)
        Quaternion as an array of shape (4,).

    Returns
    -------
    np.ndarray: Product of the two input quaternions.

    Raises
    ------
    ValueError
        If the number of columns of r is not 4.
        If the number of columns of q is not 4.
        If the number of rows of r and q are not the same.

    """

    if r.shape[1] != 4:
        raise ValueError(f"r must have 4 columns, but has {r.shape[1]}")

    if q.shape[1] != 4:
        raise ValueError(f"q must have 4 columns, but has {q.shape[1]}")

    if r.shape[0] != q.shape[0]:
        raise ValueError(
            f"r and q must have the same number of rows, but have {r.shape[0]} and {q.shape[0]} respectively")

    n0 = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    n1 = r[0] * q[1] + r[1] * q[0] + r[2] * q[3] - r[3] * q[2]
    n2 = r[0] * q[2] - r[1] * q[3] + r[2] * q[0] + r[3] * q[1]
    n3 = r[0] * q[3] + r[1] * q[2] - r[2] * q[1] + r[3] * q[0]

    n = np.array([n0, n1, n2, n3])
    return n
