"""Helper functions for reorientation of IMU data."""

import numpy as np

def acceleration(IMU: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Calculates the rotated acceleration based on the IMU data and the quaternions.

    Parameters
    ----------
    IMU
        The IMU data as an array of shape (n, 3).
    q
        The quaternions as an array of shape (n, 4).

    Returns
    -------
    The rotated acceleration as an array of shape (n, 3).

    """
    a = np.zeros((len(q), 3))
    for i in range(len(q)):
        a[i, :] = quaterot(q[i, :], IMU[i, 0:3])
    return a

def quaterot(q: np.ndarray, acc: np.ndarray, qacc=None) -> np.ndarray:
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

    """
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

    """
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

    """
    n0 = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    n1 = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    n2 = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    n3 = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]

    n = np.array([n0, n1, n2, n3])
    return n


