"""Numerical jerk calculations shared by smoothness and turn outcomes."""

import numpy as np


def _integrated_squared_derivative(signal: np.ndarray, sampling_rate_hz: float, derivative_order: int) -> np.ndarray:
    """Integrate the squared n-th time derivative of each column of ``signal``."""
    if len(signal) < 2:
        return np.full(signal.shape[1], np.nan)
    dt = 1 / sampling_rate_hz
    derivative = signal
    for _ in range(derivative_order):
        derivative = np.gradient(derivative, dt, axis=0)
    return np.trapezoid(derivative**2, dx=dt, axis=0)


def _log_dimensionless_jerk(signal: np.ndarray, sampling_rate_hz: float, derivative_order: int) -> np.ndarray:
    r"""Calculate the log dimensionless jerk for a signal.

    ``derivative_order=1`` expects an acceleration (LDLJ-A: ``T / a_peak^2 * int (da/dt)^2 dt``).
    ``derivative_order=2`` expects a velocity (LDLJ-V: ``T^3 / v_peak^2 * int (d2v/dt2)^2 dt``).

    Columns without a valid result (constant signal, too few samples, NaNs) are returned as NaN.
    """
    n_samples, n_columns = signal.shape
    result = np.full(n_columns, np.nan)
    if n_samples < 3:
        return result

    duration = (n_samples - 1) / sampling_rate_hz
    peak = np.max(np.abs(signal - signal.mean(axis=0)), axis=0)
    integral = _integrated_squared_derivative(signal, sampling_rate_hz, derivative_order)
    arg = duration ** (2 * derivative_order - 1) * integral / np.where(peak > 0, peak**2, np.nan)
    valid = np.isfinite(arg) & (arg > 1e-12)
    result[valid] = -np.log(arg[valid])
    return result
