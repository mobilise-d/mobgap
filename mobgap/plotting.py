from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import transforms
from scipy import stats

color_blind_friendly_pallet_dict = {
    "Or": (0.9, 0.6, 0),
    "SB": (0.35, 0.7, 0.9),
    "bG": (0, 0.6, 0.5),
    "Ye": (0.95, 0.9, 0.25),
    "Bu": (0, 0.45, 0.7),
    "Ve": (0.8, 0.4, 0),
    "rP": (0.8, 0.6, 0.7),
    "Bl": (0, 0, 0),
}

color_blind_friendly_pallet = list(color_blind_friendly_pallet_dict.values())


def blandaltman_stats(
    m1: pd.Series, m2: pd.Series, x_val: Literal["mean", "m1", "m2"] = "mean"
) -> tuple[pd.Series, pd.Series]:
    if x_val == "mean":
        x = (m1 + m2) / 2
    elif x_val == "m1":
        x = m1.copy()
    elif x_val == "m2":
        x = m2.copy()
    else:
        raise ValueError("x_val must be one of `mean`, `m1`, `m2`.")
    return x, m1 - m2


def plot_blandaltman_annotations(
    error: pd.Series,
    ax: Optional[plt.Axes] = None,
    agreement: float = 1.96,
    confidence: float = 0.95,
) -> plt.Axes:
    """Add annotations (mean and confidence interval) to a blandaltman style plot.

    Code modified based on penguin

    Parameters
    ----------
    error
        values typically plotted on the y-axis of the blandaltma plot
    agreement : float
        Multiple of the standard deviation to draw confidenc interval line
    confidence : float
        CIs for the limits of agreement and the mean
    ax : matplotlib axes
        Axis on which to draw the plot.

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    """
    if ax is None:
        ax = plt.gca()

    # Calculate mean, STD and SEM of x - y
    n = error.size
    dof = n - 1
    mean_diff = np.mean(error)
    std_diff = np.std(error, ddof=1)
    mean_diff_se = np.sqrt(std_diff**2 / n)
    # Limits of agreements
    high = mean_diff + agreement * std_diff
    low = mean_diff - agreement * std_diff
    high_low_se = np.sqrt(3 * std_diff**2 / n)

    # limits of agreement
    ax.axhline(mean_diff, color="k", linestyle="-", lw=2)
    ax.axhline(high, color="k", linestyle=":", lw=1.5)
    ax.axhline(low, color="k", linestyle=":", lw=1.5)

    # Annotate values
    loa_range = high - low
    offset = (loa_range / 100.0) * 1.5
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    xloc = 0.98
    ax.text(xloc, mean_diff + offset, "Mean", ha="right", va="bottom", transform=trans)
    ax.text(xloc, mean_diff - offset, f"{mean_diff:.2f}", ha="right", va="top", transform=trans)
    ax.text(xloc, high + offset, f"+{agreement:.2f} SD", ha="right", va="bottom", transform=trans)
    ax.text(xloc, high - offset, f"{high:.2f}", ha="right", va="top", transform=trans)
    ax.text(xloc, low - offset, f"-{agreement:.2f} SD", ha="right", va="top", transform=trans)
    ax.text(xloc, low + offset, f"{low:.2f}", ha="right", va="bottom", transform=trans)

    # Add 95% confidence intervals for mean bias and limits of agreement
    assert 0 < confidence < 1
    ci = {}
    ci["mean"] = stats.t.interval(confidence, dof, loc=mean_diff, scale=mean_diff_se)
    ci["high"] = stats.t.interval(confidence, dof, loc=high, scale=high_low_se)
    ci["low"] = stats.t.interval(confidence, dof, loc=low, scale=high_low_se)
    ax.axhspan(ci["mean"][0], ci["mean"][1], facecolor="tab:grey", alpha=0.2)
    ax.axhspan(ci["high"][0], ci["high"][1], facecolor="tab:blue", alpha=0.2)
    ax.axhspan(ci["low"][0], ci["low"][1], facecolor="tab:blue", alpha=0.2)

    return ax


def plot_regline(x: pd.Series, y: pd.Series, ax) -> None:
    """Plot a regression line using seaborn's regplot and add the equation to the plot as legend."""
    sns.regplot(x=x, y=y, ax=ax, scatter=False, line_kws={"color": "black", "linestyle": "--"}, truncate=True)
    # Create label
    r, p = stats.pearsonr(x, y)
    p_text = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    label = f"Regression: R = {r:.2f}, p {p_text}"

    # We add the legend as a secondary legend. I.e. we first get the original legend, save it and then add the new
    # legend and then add the old one back.
    # Get the original legend
    handles, labels = ax.get_legend_handles_labels()
    # Find the plotted line. We assume it is the last added line
    line = ax.lines[-1]
    new_legend = ax.legend([line], [label])
    # Read the original legend
    ax.legend(handles, labels)
    ax.add_artist(new_legend)


def calc_min_max_with_margin(data_x: pd.Series, data_y: pd.Series, margin: float = 0.05) -> tuple[float, float]:
    """Calculate the min and max values of a dataset with a margin."""
    data_min = min(data_x.min(), data_y.min())
    data_max = max(data_x.max(), data_y.max())
    data_range = data_max - data_min
    data_min_max = data_min - data_range * margin, data_max + data_range * margin
    return data_min_max


def make_square(ax: plt.Axes, min_max: tuple[float, float], draw_diagonal: bool = True) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(*min_max)
    ax.set_ylim(*min_max)
    if draw_diagonal:
        ax.axline((min_max[0], min_max[0]), slope=1, color="black", linestyle="-", zorder=-100)


def residual_plot(data, reference: str, detected: str, hue: str, unit: str, ax: plt.Axes) -> plt.Axes:
    """Create residual plots for method comparison using a Bland-Altman style analysis."""
    # We need to drop Na values, as the plotting functions don't handle them well
    data = data[[detected, reference, hue]].dropna(how="any")

    x, diff = blandaltman_stats(
        data[detected].astype(float),
        data[reference].astype(float),
        x_val="m2",
    )

    sns.scatterplot(x=x, y=diff, hue=data[hue], ax=ax)
    plot_blandaltman_annotations(diff, ax=ax)
    plot_regline(x, diff, ax=ax)
    ax.set_ylabel(f"WD - Reference [{unit}]")
    ax.set_xlabel(f"Reference [{unit}]")

    return ax
