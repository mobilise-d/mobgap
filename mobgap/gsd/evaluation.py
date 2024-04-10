"""Class to validate gait sequence detection results."""

from typing import Any, Optional, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from intervaltree.interval import Interval
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tpcp import OptimizablePipeline
from tpcp.validate import Aggregator, NoAgg
from typing_extensions import Self, Unpack

from mobgap.data.base import BaseGaitDatasetWithReference
from mobgap.gsd.base import BaseGsDetector, base_gsd_docfiller
from mobgap.utils.evaluation import (
    accuracy_score,
    count_samples_in_intervals,
    count_samples_in_match_intervals,
    npv_score,
    precision_recall_f1_score,
    specificity_score,
)


def calculate_matched_gsd_performance_metrics(
    matches: pd.DataFrame,
    *,
    zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[str, Union[float, int]]:
    """
    Calculate commonly known performance metrics for based on the matched overlap with the reference.

    This method assumes that you already calculated the overlapping regions between the ground truth and the detected
    gait sequences using the :func:`~mobgap.gsd.evaluation.categorize_intervals` method.
    This function then calculates the performance metrics based on the number of true positive, false positive,
    false negative, and true negative matches between detected and reference.
    All calculations are performed on the sample level.
    This means the intervals provided by the input dataframe are used to calculate the number of samples in each
    category.

    The following metrics are always returned:

    - `tp_samples`: Number of samples that are correctly detected as gait sequences.
    - `fp_samples`: Number of samples that are falsely detected as gait sequences.
    - `fn_samples`: Number of samples that are not detected as gait sequences.
    - `precision`: Precision of the detected gait sequences.
    - `recall`: Recall of the detected gait sequences.
    - `f1_score`: F1 score of the detected gait sequences.

    See the documentation of :func:`~mobgap.utils.evaluation.precision_recall_f1_score` for more details.

    Further metrics are calculated if `matches` contains true negative intervals.
    This can be achieved by passing `n_overall_samples` as additional information to
    :func:`~mobgap.gsd.evaluation.categorize_intervals`.

    - `tn_samples`: Number of samples that are correctly not detected as gait sequences.
    - `specificity`: Specificity of the detected gait sequences.
    - `accuracy`: Accuracy of the detected gait sequences.
    - `npv`: Negative predictive value of the detected gait sequences.

    See the documentation of :func:`~mobgap.utils.evaluation.specificity_score`,
     :func:`~mobgap.utils.evaluation.accuracy_score`, and :func:`~mobgap.utils.evaluation.npv_score` for more
     details.

    Parameters
    ----------
    matches: pd.DataFrame
        A DataFrame as returned by :func:`~mobgap.gsd.evaluation.categorize_intervals`.
        It contains the matched intervals between algorithm output and reference with their `start` and `end` index
        and the respective `match_type`.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    gsd_metrics: dict

    See Also
    --------
    calculate_unmatched_gsd_performance_metrics
        For calculating performance metrics without matching the detected and reference gait sequences.
    categorize_intervals
        For categorizing the detected and reference gait sequences on a sample-wise level.
        This is required to calculate the ``matches`` parameter for this function.

    """
    matches = _check_matches_sanity(matches)

    tp_samples = count_samples_in_match_intervals(matches, match_type="tp")
    fp_samples = count_samples_in_match_intervals(matches, match_type="fp")
    fn_samples = count_samples_in_match_intervals(matches, match_type="fn")
    tn_samples = count_samples_in_match_intervals(matches, match_type="tn")

    # estimate performance metrics
    precision_recall_f1 = precision_recall_f1_score(matches, zero_division=zero_division)

    gsd_metrics = {"tp_samples": tp_samples, "fp_samples": fp_samples, "fn_samples": fn_samples, **precision_recall_f1}

    # tn-dependent metrics
    if tn_samples != 0:
        gsd_metrics["tn_samples"] = tn_samples
        gsd_metrics["specificity"] = specificity_score(matches, zero_division=zero_division)
        gsd_metrics["accuracy"] = accuracy_score(matches, zero_division=zero_division)
        gsd_metrics["npv"] = npv_score(matches, zero_division=zero_division)

    return gsd_metrics


def calculate_unmatched_gsd_performance_metrics(
    *,
    gsd_list_detected: pd.DataFrame,
    gsd_list_reference: pd.DataFrame,
    sampling_rate_hz: float,
    zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[str, Union[float, int]]:
    """
    Calculate general performance metrics that don't rely on matching the detected and reference gait sequences.

    Metrics calculated by this function are just based on the overall amount of gait detected.
    The following metrics are calculated:

    - `reference_gs_duration_s`: Total duration of the reference gait sequences in seconds.
    - `detected_gs_duration_s`: Total duration of the detected gait sequences in seconds.
    - `gs_duration_error_s`: Difference between the detected and reference gait sequence duration in seconds.
    - `gs_relative_duration_error`: gs_duration_error_s divided by reference_gs_duration_s.
    - `gs_absolute_duration_error_s`: Absolute value of gs_duration_error_s.
    - `gs_absolute_relative_duration_error`: gs_absolute_duration_error_s divided by reference_gs_duration_s.
    - `gs_absolute_relative_duration_error_log`: np.log(1 + gs_duration_error_abs_rel)
    - `detected_num_gs`: Total number of detected gait sequences.
    - `reference_num_gs`: Total number of reference gait sequences.
    - `num_gs_error`: Difference between the detected and reference number of gait sequences.
    - `num_gs_relative_error`: num_gs_error divided by reference_num_gs.
    - `num_gs_absolute_error`: Absolute value of num_gs_error.
    - `num_gs_absolute_relative_error`: num_gs_absolute_error divided by reference_num_gs.
    - `num_gs_absolute_relative_error_log`: np.log(1 + num_gs_absolute_relative_error)

    Parameters
    ----------
    gsd_list_detected
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `end`.
    gsd_list_reference
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    sampling_rate_hz
        Sampling frequency of the recording in Hz.

    Returns
    -------
    gsd_metrics: dict

    See Also
    --------
    calculate_matched_gsd_performance_metrics
        For calculating performance metrics based on the matched overlap with the reference.
    categorize_intervals
        For categorizing the detected and reference gait sequences on a sample-wise level.

    """
    # estimate duration metrics
    reference_gs_duration_s = count_samples_in_intervals(gsd_list_reference) / sampling_rate_hz
    detected_gs_duration_s = count_samples_in_intervals(gsd_list_detected) / sampling_rate_hz
    gs_duration_error_s = detected_gs_duration_s - reference_gs_duration_s
    gs_relative_duration_error = np.array(gs_duration_error_s) / reference_gs_duration_s
    gs_absolute_duration_error_s = abs(gs_duration_error_s)
    gs_absolute_relative_duration_error = np.array(gs_absolute_duration_error_s) / reference_gs_duration_s
    gs_absolute_relative_duration_error_log = np.log(1 + gs_absolute_relative_duration_error)

    # estimate gs count metrics
    detected_num_gs = len(gsd_list_detected)
    reference_num_gs = len(gsd_list_reference)
    num_gs_error = detected_num_gs - reference_num_gs
    num_gs_relative_error = num_gs_error / np.array(reference_num_gs)
    num_gs_absolute_error = abs(num_gs_error)
    num_gs_absolute_relative_error = num_gs_absolute_error / np.array(reference_num_gs)
    num_gs_absolute_relative_error_log = np.log(1 + num_gs_absolute_relative_error)

    gsd_metrics = {
        "reference_gs_duration_s": reference_gs_duration_s,
        "detected_gs_duration_s": detected_gs_duration_s,
        "gs_duration_error_s": gs_duration_error_s,
        "gs_relative_duration_error": gs_relative_duration_error,
        "gs_absolute_duration_error_s": gs_absolute_duration_error_s,
        "gs_absolute_relative_duration_error": gs_absolute_relative_duration_error,
        "gs_absolute_relative_duration_error_log": gs_absolute_relative_duration_error_log,
        "detected_num_gs": detected_num_gs,
        "reference_num_gs": reference_num_gs,
        "num_gs_error": num_gs_error,
        "num_gs_relative_error": num_gs_relative_error,
        "num_gs_absolute_error": num_gs_absolute_error,
        "num_gs_absolute_relative_error": num_gs_absolute_relative_error,
        "num_gs_absolute_relative_error_log": num_gs_absolute_relative_error_log,
    }

    return gsd_metrics


def categorize_intervals(
    *, gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame, n_overall_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Evaluate detected gait sequence intervals against a reference on a sample-wise level.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.
    Each sample from the detected interval list is categorized as true positive (tp), false positive (fp),
    false negative (fn), or - if the total length of the recording (``n_overall_samples``) is provided - true negative
    (tn).
    The results are concatenated into intervals of tp, fp, fn, and tn matches and returned as a DataFrame.

    The output of this method can be used to calculate performance metrics using the
    :func:`~mobgap.gsd.evaluation.calculate_matched_gsd_performance_metrics` method.

    Parameters
    ----------
    gsd_list_detected
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `end`.
    gsd_list_reference
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    n_overall_samples
        Number of samples in the analyzed recording. If provided, true negative intervals will be added to the result.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the categorized intervals with their `start` and `end` index and the respective
        `match_type`.
        Keep in mind that the intervals are not identical to the intervals in `gsd_list_detected`, but are rather split
        into subsequences according to their match type with the reference.

    Examples
    --------
    >>> from mobgap.gsd.evaluation import categorize_intervals
    >>> detected = pd.DataFrame([[0, 10], [20, 30]], columns=["start", "end"])
    >>> reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
    >>> result = categorize_intervals(detected, reference)
    >>> result.tp_intervals
           start  end match_type
    0      0   10         tp
    1     15   20         fn
    2     20   25         tp
    3     25   30         fp

    See Also
    --------
    calculate_matched_gsd_performance_metrics
        For calculating performance metrics based on the matches returned by this function.
    calculate_unmatched_gsd_performance_metrics
        For calculating performance metrics without matching the detected and reference gait sequences.

    """
    detected, reference = _check_input_sanity(gsd_list_detected, gsd_list_reference)

    if n_overall_samples and n_overall_samples < max(gsd_list_reference["end"].max(), gsd_list_detected["end"].max()):
        raise ValueError(
            "The provided `n_samples` parameter is implausible. The number of samples must be larger than the highest "
            "end value in the detected and reference gait sequences."
        )

    # Create Interval Trees
    reference_tree = IntervalTree.from_tuples(reference.to_numpy())
    detected_tree = IntervalTree.from_tuples(detected.to_numpy())

    # Prepare DataFrames for TP, FP, FN
    tp_intervals = []
    fp_intervals = []
    fn_intervals = []

    # Calculate TP and FP
    for interval in detected_tree:
        overlaps = sorted(reference_tree.overlap(interval.begin, interval.end))
        if overlaps:
            fp_matches = _get_false_matches_from_overlap_data(overlaps, interval)
            fp_intervals.extend(fp_matches)

            for overlap in overlaps:
                start = max(interval.begin, overlap.begin)
                end = min(interval.end, overlap.end)
                tp_intervals.append([start, end])

        else:
            fp_intervals.append([interval.begin, interval.end])

    # Calculate FN
    for interval in reference_tree:
        overlaps = sorted(detected_tree.overlap(interval.begin, interval.end))
        if not overlaps:
            fn_intervals.append([interval.begin, interval.end])
        else:
            fn_matches = _get_false_matches_from_overlap_data(overlaps, interval)
            fn_intervals.extend(fn_matches)

    # convert results to pandas DataFrame and add a match type column
    tp_intervals = pd.DataFrame(tp_intervals, columns=["start", "end"])
    tp_intervals["match_type"] = "tp"
    fp_intervals = pd.DataFrame(fp_intervals, columns=["start", "end"])
    fp_intervals["match_type"] = "fp"
    fn_intervals = pd.DataFrame(fn_intervals, columns=["start", "end"])
    fn_intervals["match_type"] = "fn"

    categorized_intervals = pd.concat([tp_intervals, fp_intervals, fn_intervals], ignore_index=True)
    categorized_intervals = categorized_intervals.sort_values(by=["start", "end"], ignore_index=True)

    # add tn intervals
    if n_overall_samples is not None:
        tn_intervals = _get_tn_intervals(categorized_intervals, n_overall_samples=n_overall_samples)
        categorized_intervals = pd.concat([categorized_intervals, tn_intervals], ignore_index=True)
        categorized_intervals = categorized_intervals.sort_values(by=["start", "end"], ignore_index=True)

    return categorized_intervals


def _check_matches_sanity(matches: pd.DataFrame) -> pd.DataFrame:
    # check if input is a dataframe
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("`matches` must be of type `pandas.DataFrame`.")
    # check for correct columns
    try:
        matches = matches[["start", "end", "match_type"]]
    except KeyError as e:
        raise ValueError("`matches` must have columns named `start`, `end`, and `match_type`.") from e
    # check if `match_type` column contains only valid values
    if not matches["match_type"].isin(["tp", "fp", "fn", "tn"]).all():
        raise ValueError("`match_type` must contain only the values 'tp', 'fp', 'fn', and 'tn'.")
    return matches


def _check_input_sanity(
    gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # check if inputs are dataframes
    if not isinstance(gsd_list_detected, pd.DataFrame) or not isinstance(gsd_list_reference, pd.DataFrame):
        raise TypeError("`gsd_list_detected` and `gsd_list_reference` must be of type `pandas.DataFrame`.")
    # check if start and end columns are present
    try:
        detected, reference = gsd_list_detected[["start", "end"]], gsd_list_reference[["start", "end"]]
    except KeyError as e:
        raise ValueError(
            "`gsd_list_detected` and `gsd_list_reference` must have columns named 'start' and 'end'."
        ) from e
    return detected, reference


def _get_false_matches_from_overlap_data(overlaps: list[Interval], interval: Interval) -> list[list[int]]:
    f_intervals = []
    for i, overlap in enumerate(overlaps):
        prev_el = overlaps[i - 1] if i > 0 else None
        next_el = overlaps[i + 1] if i < len(overlaps) - 1 else None

        # check if there are false matches before the overlap
        if interval.begin < overlap.begin:
            fn_start = interval.begin
            # check if interval is already covered by a previous overlap
            if prev_el and interval.begin < prev_el.end:
                fn_start = prev_el.end
            f_intervals.append([fn_start, overlap.begin])

        # check if there are false matches after the overlap
        if interval.end > overlap.end:
            fn_end = interval.end
            # check if interval is already covered by a succeeding overlap
            if next_el and interval.end > next_el.begin:
                # skip because this will be handled by the next iteration
                continue
                # fn_end = next_el.begin
            f_intervals.append([overlap.end, fn_end])

    return f_intervals


def find_matches_with_min_overlap(
    *, gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame, overlap_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Find all matches of `gsd_list_detected` in `gsd_list_reference` with at least ``overlap_threshold`` overlap.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.

    Note, that the threshold is enforced in both directions. That means, that the relative overlap of the detected gait
    sequence with respect to the overall length of the detected interval AND to the overall length of the matched
    reference interval must be at least `overlap_threshold`.

    Note, we assume that ``gsd_list_detected`` has no overlaps, but we don't enforce it!

    Parameters
    ----------
    gsd_list_detected: pd.DataFrame
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in the first and the stop index in the second column.
       Furthermore, the id of the respective gait sequence can be provided in the third column.
    gsd_list_reference: pd.DataFrame
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    overlap_threshold: float
        The minimum relative overlap between a detected sequence and its reference with respect to the length of both
        intervals.
        Must be larger than 0.5 and smaller than or equal to 1.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the intervals from `gsd_list_detected` that overlap with `gsd_list_reference` with the
        specified minimum overlap.
        The dataframe contains the gait sequence ids as index column as well as the start and end indices of the
        intervals.

    Examples
    --------
    >>> from mobgap.gsd.evaluation import find_matches_with_min_overlap
    >>> detected = pd.DataFrame([[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> reference = pd.DataFrame([[0, 10, 0], [15, 25, 1]], columns=["start", "end", "id"]).set_index("id")
    >>> result = find_matches_with_min_overlap(detected, reference)
       start  end
    id
    0      0   10

    """
    detected, reference = _check_input_sanity(gsd_list_detected, gsd_list_reference)

    # check if index is unique
    if not gsd_list_detected.index.is_unique:
        raise ValueError("`gsd_list_detected` must have a unique index.")

    if overlap_threshold <= 0.5:
        raise ValueError(
            "overlap_threshold must be greater than 0.5."
            "Otherwise multiple matches between intervals "
            "are possible."
        )
    if overlap_threshold > 1:
        raise ValueError("overlap_threshold must be less than 1." "Otherwise no matches can be returned.")

    tree = IntervalTree.from_tuples(detected.reset_index(names="id")[["start", "end", "id"]].to_numpy())
    final_matches = []

    for _, interval in reference[["start", "end"]].iterrows():
        matches = tree[interval["start"] : interval["end"]]
        if len(matches) > 0:
            for match in matches:
                # First calculate the absolute overlap
                absolute_overlap = match.overlap_size(interval["start"], interval["end"])
                # Then calculate the relative overlap
                relative_overlap_interval = absolute_overlap / (interval["end"] - interval["start"])
                relative_overlap_match = absolute_overlap / (match[1] - match[0])
                if relative_overlap_interval >= overlap_threshold and relative_overlap_match >= overlap_threshold:
                    final_matches.append(list(match)[:3])
                    break

    final_matches = pd.DataFrame(final_matches, columns=["start", "end", "id"]).set_index("id")
    return final_matches


def _get_tn_intervals(categorized_intervals: pd.DataFrame, n_overall_samples: Union[int, None]) -> pd.DataFrame:
    """Add true negative intervals to the categorized intervals by inferring them from the other intervals.

    This function requires sorted and non-overlapping intervals in `categorized_intervals`.
    If `n_overall_samples` is not provided, an empty DataFrame is returned.
    """
    if n_overall_samples is None:
        return pd.DataFrame(columns=["start", "end", "match_type"])

    # add tn intervals
    tn_intervals = []
    for i, (start, _) in enumerate(categorized_intervals[["start", "end"]].itertuples(index=False)):
        if i == 0:
            if start > 0:
                tn_intervals.append([0, start])
        elif start > categorized_intervals.iloc[i - 1]["end"]:
            tn_intervals.append([categorized_intervals.iloc[i - 1]["end"], start])

    if categorized_intervals.iloc[-1]["end"] < n_overall_samples - 1:
        tn_intervals.append([categorized_intervals.iloc[-1]["end"], n_overall_samples - 1])

    tn_intervals = pd.DataFrame(tn_intervals, columns=["start", "end"])
    tn_intervals["match_type"] = "tn"
    return tn_intervals


def plot_categorized_intervals(
    gsd_list_detected: pd.DataFrame, gsd_list_reference: pd.DataFrame, categorized_intervals: pd.DataFrame
) -> Figure:
    """Plot the categorized intervals together with the detected and reference intervals."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _plot_intervals_from_df(gsd_list_reference, 3, ax, color="orange")
    _plot_intervals_from_df(gsd_list_detected, 2, ax, color="blue")
    _plot_intervals_from_df(categorized_intervals.query("match_type == 'tp'"), 1, ax, color="green", label="TP")
    _plot_intervals_from_df(categorized_intervals.query("match_type == 'fp'"), 1, ax, color="red", label="FP")
    _plot_intervals_from_df(categorized_intervals.query("match_type == 'fn'"), 1, ax, color="purple", label="FN")
    plt.yticks([1, 2, 3], ["Categorized", "Detected", "Reference"])
    plt.ylim(0, 4)
    plt.xlabel("Index")
    leg = plt.legend(loc="upper right", bbox_to_anchor=(1, 1.2), ncol=3, frameon=False)
    for handle in leg.legend_handles:
        handle.set_linewidth(10)
    plt.tight_layout()
    return fig


def _plot_intervals_from_df(df: pd.DataFrame, y: int, ax: Axes, **kwargs: Unpack[dict[str, Any]]) -> None:
    label_set = False
    for _, row in df.iterrows():
        label = kwargs.pop("label", None)
        if label and not label_set:
            ax.hlines(y, row["start"], row["end"], lw=20, label=label, **kwargs)
            label_set = True
        else:
            ax.hlines(y, row["start"], row["end"], lw=20, **kwargs)


@base_gsd_docfiller
class GsdEvaluationPipeline(OptimizablePipeline[BaseGaitDatasetWithReference]):
    """Pipeline that can be used during evaluation of Gsd algorithms.

    This wraps any GSD algorithm and allows to apply it to a single datapoint of a Gait Dataset or optimize it
    based on a whole dataset.

    This pipeline can be used in combination with the ``tpcp.validate`` and ``tpcp.optimize`` modules to evaluate and
    improve the performance of a GSD algorithm.

    Parameters
    ----------
    algo
        The GSD algorithm that should be run/evaluated.

    Attributes
    ----------
    %(gs_list_)s
    algo_
        The GSD algo instance with all results after running the algorithm.
        This can be helpful for debugging or further analysis.

    """

    algo: BaseGsDetector

    algo_: BaseGsDetector

    def __init__(self, algo: BaseGsDetector) -> None:
        self.algo = algo

    @property
    def gs_list_(self) -> pd.DataFrame:  # noqa: D102
        return self.algo_.gs_list_

    def run(self, datapoint: BaseGaitDatasetWithReference) -> Self:
        """Run the pipeline on a single data point.

        This extracts the imu_data (``data_ss``) and the sampling rate (``sampling_rate_hz``) from the datapoint and
        uses the ``detect`` method of the GSD algorithm to detect the gait sequences.

        Parameters
        ----------
        datapoint
            A single datapoint of a Gait Dataset with reference information.

        Returns
        -------
        self
            The pipeline instance with the detected gait sequences stored in the ``gs_list_`` attribute.

        """
        single_sensor_imu_data = datapoint.data_ss
        sampling_rate_hz = datapoint.sampling_rate_hz

        self.algo_ = self.algo.clone().detect(single_sensor_imu_data, sampling_rate_hz=sampling_rate_hz)

        return self

    def self_optimize(self, dataset: BaseGaitDatasetWithReference, **kwargs: Unpack[dict[str, Any]]) -> Self:
        """Run a "self-optimization" of a GSD-algorithm (if it implements the respective method).

        This method extracts all data and reference gait sequences from the dataset and uses them to optimize the
        algorithm by calling the ``self_optimize`` method of the GSD algorithm (if it is implemented).

        Note, that this is only useful for algorithms with "internal" optimization logic (i.e. ML-based algorithms).
        If you want to optimize the hyperparameters of the algorithm, you should use the ``tpcp.optimize`` module.

        Parameters
        ----------
        dataset
            A Gait Dataset with reference information.
        kwargs
            Additional parameters required for the optimization process.
            This will be passed to the ``self_optimize`` method of the GSD algorithm.

        Returns
        -------
        self
            The pipeline instance with the optimized GSD algorithm.

        """
        all_data = [d.data_ss for d in dataset]
        reference_wbs = [d.reference_parameters_.wb_list for d in dataset]
        sampling_rate_hz = [d.sampling_rate_hz for d in dataset]

        self.algo.self_optimize(all_data, reference_wbs, sampling_rate_hz=sampling_rate_hz, **kwargs)

        return self

    def score(
        self, datapoint: BaseGaitDatasetWithReference
    ) -> Union[float, dict[str, Union[float, Aggregator]], Aggregator]:
        """Score the performance of the GSD algorithm of a single datapoint using standard metrics.

        This method applies the GSD algorithm to the datapoint (using the implemented ``run`` method) and calculates
        the performance metrics based on the detected gait sequences and the reference gait sequences.

        This method can be used as a scoring function in the ``tpcp.validate`` module.

        Parameters
        ----------
        datapoint
            A single datapoint of a Gait Dataset with reference information.

        Returns
        -------
        dict
            Dictionary of performance metrics.

        See Also
        --------
        calculate_matched_gsd_performance_metrics
            For calculating performance metrics based on the matched overlap with the reference.
        calculate_unmatched_gsd_performance_metrics
            For calculating performance metrics without matching the detected and reference gait sequences.
        categorize_intervals
            For categorizing the detected and reference gait sequences on a sample-wise level.

        """
        detected_gs_list = self.safe_run(datapoint).gs_list_
        reference_gs_list = datapoint.reference_parameters_.wb_list[["start", "end"]]
        n_datapoints = len(datapoint.data_ss)
        sampling_rate_hz = datapoint.sampling_rate_hz

        matches = categorize_intervals(
            gsd_list_detected=detected_gs_list, gsd_list_reference=reference_gs_list, n_overall_samples=n_datapoints
        )

        all_metrics = {
            **calculate_unmatched_gsd_performance_metrics(
                gsd_list_detected=detected_gs_list,
                gsd_list_reference=reference_gs_list,
                sampling_rate_hz=sampling_rate_hz,
            ),
            **calculate_matched_gsd_performance_metrics(matches),
            "detected": NoAgg(detected_gs_list),
            "reference": NoAgg(reference_gs_list),
        }

        return all_metrics


__all__ = [
    "categorize_intervals",
    "find_matches_with_min_overlap",
    "calculate_matched_gsd_performance_metrics",
    "calculate_unmatched_gsd_performance_metrics",
    "plot_categorized_intervals",
    "GsdEvaluationPipeline",
]
