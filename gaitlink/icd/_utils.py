import matplotlib.pyplot as plt
import numpy as np


def zerocros(y, m="b", x=None):
    if x is None:
        x = np.arange(len(y)) + 1  # Default x-axis values

    # Check if 'm' is provided, if not, set it to the default 'b' (both)
    if not m or len(m) == 0:
        m = "b"

    # Create a boolean array s where True indicates positive values in y
    s = (y >= 0).astype(int)

    # Calculate the differences between consecutive elements of s to identify crossings.
    k = np.diff(s)

    # Determine the indices (f) of zero crossings based on the provided mode string (m)
    if "p" in m:
        f = np.where(k > 0)[0]  # positive only
    elif "n" in m:
        f = np.where(k < 0)[0]  # negative only
    else:
        f = np.where(k != 0)[0]  # Find both

    # Estimate the slope (s) of y at the zero crossing points.
    s = y[f + 1] - y[f]

    # Estimate the x-axis positions (t) of the zero crossings using linear interpolation.
    t = f - y[f] / s

    # If the 'r' (round to sample values) mode is specified, round the positions.
    if "r" in m:
        t = np.round(t)

    # If x-axis values (x) are provided, refine t and s using interpolation.
    if x is not None:
        tf = t - f  # Fractional sample position.
        t = x[f] * (1 - tf) + x[f + 1] * tf
        s = s / (x[f + 1] - x[f])

    # TODO: check plots
    # Plot the signal and zero crossing points if no return is requested.
    if not t.any() or not s.any():
        n = len(y)
        if x is not None:
            plt.plot(x, y, "-b", t, np.zeros(len(t), dtype=float), "or")
        else:
            plt.plot(np.arange(1, n + 1), y, "-b", t, np.zeros(len(t), dtype=float), "or")
        plt.xlim([-1, -1.05])  # Enlarge the axis for visualization.

    return t, s
