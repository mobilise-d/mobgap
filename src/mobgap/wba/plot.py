"""Plot functions to visualize the results of a Walking Bout Assembly and Stride Selection."""

from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from mobgap.wba import StrideSelection, WbAssembly


def plot_wba_results(
    wba: WbAssembly, stride_selection: Optional[StrideSelection] = None, *, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot the results of a Walking Bout Assembly (and optionally a stride selection).

    Parameters
    ----------
    wba
        The Walking Bout Assembly object with results attached (i.e. after calling `assemble`).
    stride_selection
        The stride selection object with results attached (i.e. after calling `filter`).
        This is optional.
    ax
        The matplotlib axes to plot on.
        If `None` a new figure and axes are created.

    Returns
    -------
    ax
        The matplotlib axes that were used for plotting.

    """
    try:
        stride_list = wba.annotated_stride_list_
        excluded_stride_list = wba.excluded_stride_list_
    except AttributeError as e:
        raise ValueError("The WbAssembly object does not contain a stride list. Please run `assemble` first.") from e

    excluded_stride_list["excluded"] = True
    stride_list["excluded"] = False
    combined_stride_list = pd.concat([stride_list, excluded_stride_list])
    # We plot each stride as a small box based on start and end time.
    # Excluded strides are plotted in red.
    # We plot left and right strides in two rows in the same plot, as they overlap otherwise.

    if ax is None:
        _, ax = plt.subplots(figsize=(20, 10))

    y_tick_positions = []
    y_tick_labels = []

    stride_row_centers = (2, 3)
    rectangles, feet_label = _plot_stride_list(combined_stride_list, stride_row_centers, 1)
    y_tick_labels.extend(feet_label)
    y_tick_positions.extend(stride_row_centers)
    ax.add_collection(PatchCollection(rectangles, match_original=True))

    # Finally we plot the WBs
    wb_row = 1
    wb_box_height = 0.5
    wb_box_half_height = wb_box_height / 2

    y_tick_positions.append(wb_row)
    y_tick_labels.append("WB")

    def _plot_wb(start: float, end: float, color: str) -> None:
        ax.add_patch(
            Rectangle(
                (start, wb_row - wb_box_half_height),
                end - start,
                wb_box_height,
                facecolor=color,
                edgecolor="black",
                alpha=0.3,
            )
        )

    for _, stride_list in wba.wbs_.items():
        _plot_wb(stride_list.iloc[0].start, stride_list.iloc[-1].end, "blue")

    for _, stride_list in wba.excluded_wbs_.items():
        _plot_wb(stride_list.iloc[0].start, stride_list.iloc[-1].end, "red")

    if stride_selection:
        try:
            stride_list = stride_selection.filtered_stride_list_
            excluded_stride_list = stride_selection.excluded_stride_list_
        except AttributeError as e:
            raise ValueError(
                "The StrideSelection object does not contain a stride list. Please run `filter` first."
            ) from e

        # We check that the stride selection object belongs to the WBA object.
        if not wba.filtered_stride_list.equals(stride_list):
            raise ValueError(
                "It seems like the WBAssembly and StrideSelection objects do not belong together. "
                "You can only plot stride selection and wba results together, when you called the WBA with the "
                "`filtered_stride_list_` output of the stride selection."
            )

        excluded_stride_list["excluded"] = True
        stride_list["excluded"] = False
        combined_stride_list = pd.concat([stride_list, excluded_stride_list])

        ss_row_centers = (4.25, 5.25)
        rectangles, feet_label = _plot_stride_list(combined_stride_list, ss_row_centers, 1)
        y_tick_labels.extend(feet_label)
        y_tick_positions.extend(ss_row_centers)

    ax.add_collection(PatchCollection(rectangles, match_original=True))

    # we set a custom y-axis
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    # We calculate the stride list of one stride to get an idea what unit the x-axis is in.
    approx_stride_duration = combined_stride_list.iloc[0].end - combined_stride_list.iloc[0].start

    ax.set_xlim(
        combined_stride_list["start"].min() - approx_stride_duration,
        combined_stride_list["end"].max() + approx_stride_duration,
    )
    ax.set_ylim(min(y_tick_positions) - 0.5, max(y_tick_positions) + 0.75)

    return ax


def _plot_stride_list(
    stride_list: pd.DataFrame, row_centers: tuple[float, float], box_height: float
) -> tuple[list[Rectangle], list[str]]:
    box_half_height = box_height / 2
    rectangles = []
    feet_labels = []
    for (foot, strides), row_center in zip(stride_list.groupby("foot"), row_centers):
        for stride in strides.itertuples(index=False):
            rectangles.append(
                Rectangle(
                    (stride.start, row_center - box_half_height),
                    stride.end - stride.start,
                    box_height,
                    facecolor="red" if stride.excluded else "green",
                    edgecolor="black",
                    alpha=0.3,
                )
            )
        feet_labels.append(foot)
    return rectangles, feet_labels


__all__ = ["plot_wba_results"]
