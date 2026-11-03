"""Interval containment, matching, counting, conversion, and filtering utilities."""

import warnings
from heapq import heappush, heappushpop
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from intervaltree.interval import Interval

from mobgap._gaitmap.utils.array_handling import (
    bool_array_to_start_end_array,
    merge_intervals,
    start_end_array_to_bool_array,
)


def map_intervals_to_regions(sequences: pd.DataFrame, regions: pd.DataFrame) -> pd.Series:
    """Find the region that fully contains each sequence.

    Parameters
    ----------
    sequences
        Dataframe with ``start`` and ``end`` columns and unique row identifiers.
        Sequences may overlap and need not be sorted.
    regions
        Dataframe with ``start`` and ``end`` columns and unique row identifiers.
        Region identifiers may be arbitrary hashable values, including MultiIndex tuples.
        Both tables must use the same coordinate system, with nonmissing coordinates
        and ``start <= end``. Intervals are half-open, but equal boundaries count as
        containment: ``region.start <= sequence.start`` and ``sequence.end <= region.end``.

    Returns
    -------
    pandas.Series
        Object Series with the index and row order of ``sequences``. Each value is
        a region identifier, or ``pd.NA`` when no region fully contains the sequence.
        The Series has no name.

    Raises
    ------
    ValueError
        If multiple regions fully contain any sequence.

    Notes
    -----
    Matching uses coordinates only, never index labels. Neither input is modified.
    Sorting followed by a sweep avoids a Cartesian overlap matrix. Of the regions
    that start before a query, only the two greatest ends are needed to distinguish
    missing, unique, and ambiguous containment.
    """
    region_rows = sorted(
        (start, end, position)
        for position, (start, end) in enumerate(regions[["start", "end"]].itertuples(index=False, name=None))
    )
    sequence_rows = sorted(
        (start, end, position)
        for position, (start, end) in enumerate(sequences[["start", "end"]].itertuples(index=False, name=None))
    )
    result = [pd.NA] * len(sequences)
    greatest_ends = []
    region_position = 0
    for start, end, position in sequence_rows:
        while region_position < len(region_rows) and region_rows[region_position][0] <= start:
            _, region_end, original_position = region_rows[region_position]
            candidate = (region_end, original_position)
            if len(greatest_ends) < 2:
                heappush(greatest_ends, candidate)
            else:
                heappushpop(greatest_ends, candidate)
            region_position += 1
        if len(greatest_ends) == 2 and greatest_ends[0][0] >= end:
            raise ValueError(f"Sequence {sequences.index[position]!r} is fully contained in multiple regions.")
        if greatest_ends:
            region_end, original_position = max(greatest_ends)
            if region_end >= end:
                result[position] = regions.index[original_position]
    return pd.Series(result, index=sequences.index, dtype=object)


def categorize_intervals_per_sample(
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
    :func:`~mobgap.gait_sequences.evaluation.calculate_matched_gsd_performance_metrics` method.

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
    >>> from mobgap.utils.intervals import categorize_intervals_per_sample
    >>> detected = pd.DataFrame([[0, 10], [20, 30]], columns=["start", "end"])
    >>> reference = pd.DataFrame([[0, 10], [15, 25]], columns=["start", "end"])
    >>> result = categorize_intervals_per_sample(detected, reference)
    >>> result.tp_intervals
           start  end match_type
    0      0   10         tp
    1     15   20         fn
    2     20   25         tp
    3     25   30         fp

    See Also
    --------
    mobgap.gait_sequences.evaluation.calculate_matched_gsd_performance_metrics
        For calculating performance metrics based on the matches returned by this function.
    mobgap.gait_sequences.evaluation.calculate_unmatched_gsd_performance_metrics
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


def _get_tn_intervals(categorized_intervals: pd.DataFrame, n_overall_samples: Union[int, None]) -> pd.DataFrame:
    """Add true negative intervals to the categorized intervals by inferring them from the other intervals.

    This function requires sorted and non-overlapping intervals in `categorized_intervals`.
    If `n_overall_samples` is not provided, an empty DataFrame is returned.
    """
    if n_overall_samples is None:
        return pd.DataFrame(columns=["start", "end", "match_type"])

    if len(categorized_intervals) == 0:
        return pd.DataFrame([[0, n_overall_samples - 1, "tn"]], columns=["start", "end", "match_type"])

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


def _sanitize_index(ic_list: pd.DataFrame, list_type: Literal["detected", "reference"]) -> tuple[pd.DataFrame, bool]:
    is_multindex = False
    # check if index is a multiindex and raise warning if it is
    if isinstance(ic_list.index, pd.MultiIndex):
        is_multindex = True
        ic_list.index = ic_list.index.to_flat_index()
        ic_list.index.name = f"gs_id_{list_type}"
    # check if indices are unique
    if not ic_list.index.is_unique:
        raise ValueError(f"The index of `gs_list_{list_type}` must be unique!")
    return ic_list, is_multindex


def categorize_intervals(
    *,
    gsd_list_detected: pd.DataFrame,
    gsd_list_reference: pd.DataFrame,
    overlap_threshold: float = 0.8,
    multiindex_warning: bool = True,
) -> pd.DataFrame:
    """Evaluate a gait sequence list against a reference sequence-by-sequence with a minimum overlap threshold.

    This compares a gait sequence list against a reference list and classifies each detected sequence as true positive,
    false positive, or false negative.
    A gait sequence is classified as true positive when having at least ``overlap_threshold`` overlap with a reference
    sequence. If a detected sequence has no overlap with any reference sequence, it is classified as false positive.
    If a reference sequence has no overlap with any detected sequence, it is classified as false negative.

    Note, that the threshold is enforced in both directions. That means, that the relative overlap of the detected gait
    sequence with respect to the overall length of the detected interval AND to the overall length of the matched
    reference interval must be at least `overlap_threshold`.

    The detected and reference dataframes are expected to have columns namend "start" and "end" containing the
    start and end indices of the respective gait sequences.
    As index, we support either a single or a multiindex without duplicates
    (i.e., the index must identify each gait sequence uniquely).
    If a multiindex is provided, the single index levels will be ignored for the comparison and matches across different
    index groups will be possible.
    If this is not the intended use case, consider grouping your input data before calling this function
    (see :func:`~mobgap.utils.array_handling.create_multi_groupby`).

    Note, we assume that ``gsd_list_detected`` has no overlaps, but we don't enforce it!
    Additionally, note that this method won't return any new intervals
    (as done in :func:`~mobgap.utils.intervals.categorize_intervals_per_sample`).
    Instead, the comparison is done on a sequence-by-sequence level based on the provided intervals.

    Parameters
    ----------
    gsd_list_detected: pd.DataFrame
       Each row contains a detected gait sequence interval as output from the GSD algorithms.
       The respective start index is stored in a column named `start` and the stop index in a column named `stop`.
    gsd_list_reference: pd.DataFrame
       Gold standard to validate the detected gait sequences against.
       Should have the same format as `gsd_list_detected`.
    overlap_threshold: float
        The minimum relative overlap between a detected sequence and its reference with respect to the length of both
        intervals.
        Must be larger than 0.5 and smaller than or equal to 1.
    multiindex_warning
        If True, a warning will be raised if the index of the input data is a MultiIndex, explaining that the index
        levels will be ignored for the matching process.
        This exists, as this is a common source of error, when this function is used together with a typical pipeline
        that iterates over individual gait sequences during the processing using :class:`~mobgap.pipeline.GsIterator`.
        Only set this to False, once you understand the two different usecases.

    Returns
    -------
    matches: pandas.DataFrame
        A 3 column dataframe with the column names `gsd_id_detected`,
        `gsd_id_reference`, and `match_type`.
        Each row is a match containing the index value of the detected and the reference list, that belong together,
        or a tuple of index values in case of a multiindex input.
        The `match_type` column indicates the type of match.
        For all gait sequences that have a match in the reference list, this will be "tp" (true positive).
        Gait sequences that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positive).
        All reference gait sequences that do not have a counterpart in the detected list
        are marked as "fn" (false negative).

    Examples
    --------
    >>> from mobgap.utils.intervals import categorize_intervals
    >>> detected = pd.DataFrame(
    ...     [[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
    >>> reference = pd.DataFrame(
    ...     [[0, 10, 0], [15, 25, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
    >>> result = categorize_intervals(detected, reference)
       gsd_id_detected  gs_id_reference match_type
    0               0               0         tp
    1               1               NaN       fp
    2               NaN             1         fn
    """
    detected, reference = _check_input_sanity(gsd_list_detected, gsd_list_reference)
    detected, is_multindex_detected = _sanitize_index(detected, "detected")
    reference, is_multindex_reference = _sanitize_index(reference, "reference")

    if multiindex_warning and (is_multindex_detected or is_multindex_reference):
        warnings.warn(
            "The index of `gsd_list_detected` or `gsd_list_reference` is a MultiIndex. "
            "Please be aware that the index levels will not be regarded separately for the matching process, "
            "and gait sequences might be matched across different index groups, such as recording sessions or "
            "participants.\n"
            "If this is not the intended use case for you, consider grouping your input data before calling the "
            "evaluation function.\n\n"
            "This can be done using the `create_multi_groupby` function from the `mobgap.utils.array_handling`. ",
            stacklevel=1,
        )

    if overlap_threshold <= 0.5:
        raise ValueError(
            "overlap_threshold must be greater than 0.5.Otherwise multiple matches between intervals are possible."
        )
    if overlap_threshold > 1:
        raise ValueError("overlap_threshold must be less than 1.Otherwise no matches can be returned.")

    tree = IntervalTree.from_tuples(
        detected.reset_index(drop=True)
        .rename_axis("range_index")
        .reset_index()[["start", "end", "range_index"]]
        .to_numpy()
    )
    detected_index = []
    reference_index = []

    for ref_id, interval in reference.reset_index()[["start", "end"]].iterrows():
        matches = tree[interval["start"] : interval["end"]]
        if len(matches) > 0:
            for match in matches:
                det_idx = list(match)[2]
                # First calculate the absolute overlap
                absolute_overlap = match.overlap_size(interval["start"], interval["end"])
                # Then calculate the relative overlap
                relative_overlap_interval = absolute_overlap / (interval["end"] - interval["start"])
                relative_overlap_match = absolute_overlap / (match[1] - match[0])
                if relative_overlap_interval >= overlap_threshold and relative_overlap_match >= overlap_threshold:
                    detected_index.append(det_idx)
                    reference_index.append(ref_id)
                    break

    detected_index_name = "gs_id_detected"
    reference_index_name = "gs_id_reference"

    matches_detected = pd.DataFrame(index=detected.index.copy(), columns=[reference_index_name])
    matches_detected.index.name = detected_index_name

    matches_reference = pd.DataFrame(index=reference.index.copy(), columns=[detected_index_name])
    matches_reference.index.name = reference_index_name

    ic_list_detected_idx = detected.iloc[detected_index].index
    ic_list_reference_idx = reference.iloc[reference_index].index

    matches_detected.loc[ic_list_detected_idx, reference_index_name] = ic_list_reference_idx
    matches_reference.loc[ic_list_reference_idx, detected_index_name] = ic_list_detected_idx

    matches_detected = matches_detected.reset_index()
    matches_reference = matches_reference.reset_index()

    matches = (
        pd.concat([matches_detected, matches_reference])
        .drop_duplicates()
        .sort_values([detected_index_name, reference_index_name])
        .reset_index(drop=True)
    )

    matches.index.name = "match_id"
    if matches.empty:
        # return empty dataframe with the correct column names
        matches.loc[:, "match_type"] = pd.Series()
        return matches

    matches.loc[~matches.isna().any(axis=1), "match_type"] = "tp"
    matches.loc[matches[reference_index_name].isna(), "match_type"] = "fp"
    matches.loc[matches[detected_index_name].isna(), "match_type"] = "fn"

    return matches


def get_matching_intervals(
    *, metrics_detected: pd.DataFrame, metrics_reference: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract the detected and reference gait sequences that are considered as matches sequence-by-sequence.

    Additionally, the metrics of the detected and reference gait sequences are extracted and returned in a DataFrame
    for further comparison.
    When your metrics are already aggregated on a higher level, such as daily, participant-wise, or session-wise,
    refer to ~func:`~mobgap.gait_sequences.evaluation.combine_det_with_ref_without_matching` instead.

    Parameters
    ----------
    metrics_detected
       Each row corresponds to a detected gait sequence interval as output from the GSD algorithms.
       The columns contain the metrics estimated for each respective gait sequence based on these detected intervals.
       The columns present in both `metrics_detected` and `metrics_reference` are regarded for the matching,
       while the other columns are discarded.
    metrics_reference
       Each row corresponds to a reference gait sequence interval as retrieved from the reference system.
       The columns contain the metrics estimated for each respective gait sequence based on these reference intervals.
       The columns present in both `metrics_detected` and `metrics_reference` are regarded for the matching,
       while the other columns are discarded.
    matches
        A DataFrame containing the matched gait sequences
        as output by :func:`~mobgap.gait_sequences.evaluation.find_matches_with_min_overlap`.
        Must have been calculated based on the same interval data as `metrics_detected` and `metrics_reference`.
        Expected to have the columns `gs_id_detected`, `gs_id_reference`, and `match_type`.

    Returns
    -------
    matches: pd.DataFrame
        The detected gait sequences that are considered as matches assigned to the reference sequences
        they are matching with.
        As index, the unique identifier for each matched gait sequence assigned in the `matches` DataFrame is used.
        The columns are two-level MultiIndex columns, consisting of a `metrics` and an `origin` level.
        As first column level, all columns present in both `metrics_detected` and `metrics_reference` are included.
        The second column level indicates the origin of the respective value, either `detected` or `reference` for
        metrics that were estimated based on the detected or reference gait sequences, respectively.

    Examples
    --------
    >>> from mobgap.utils.intervals import categorize_intervals
    >>> detected = pd.DataFrame(
    ...     [[0, 10, 0], [20, 30, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
    >>> reference = pd.DataFrame(
    ...     [[0, 10, 0], [21, 29, 1]], columns=["start", "end", "id"]
    ... ).set_index("id")
    >>> detected_metrics = pd.DataFrame(
    ...     [[1, 2, 0], [1, 2, 1]], columns=["metric_1", "metric_2", "id"]
    ... ).set_index("id")
    >>> reference_metrics = pd.DataFrame(
    ...     [[2, 3, 0], [2, 3, 1]], columns=["metric_1", "metric_2", "id"]
    ... ).set_index("id")
    >>> matches = categorize_intervals(
    ...     gsd_list_detected=detected, gsd_list_reference=reference
    ... )
    >>> matched_gs = get_matching_intervals(
    ...     metrics_detected=detected_metrics,
    ...     metrics_reference=reference_metrics,
    ...     matches=matches,
    ... )
        metric metric_1           metric_2
        origin detected reference detected reference
    id
    0             1         2        2         3
    1             1         2        2         3

    """
    matches = _check_gs_level_matches_sanity(matches)

    tp_matches = matches.query("match_type == 'tp'")

    detected_matches = extract_tp_matches(metrics_detected, tp_matches["gs_id_detected"])
    reference_matches = extract_tp_matches(metrics_reference, tp_matches["gs_id_reference"])

    combined_matches = combine_detected_and_reference_metrics(
        detected_matches, reference_matches, tp_matches=tp_matches
    )

    return combined_matches


def _check_gs_level_matches_sanity(matches: pd.DataFrame) -> pd.DataFrame:
    # check if input is a dataframe
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("`matches` must be of type `pandas.DataFrame`.")
    # check for correct columns
    try:
        matches = matches[["gs_id_detected", "gs_id_reference", "match_type"]]
    except KeyError as e:
        raise ValueError(
            "`matches` must have columns named `gs_id_detected`, `gs_id_reference`, and `match_type`."
        ) from e
    # check if `match_type` column contains only valid values
    if not matches["match_type"].isin(["tp", "fp", "fn"]).all():
        raise ValueError("`match_type` must contain only the values 'tp', 'fp', and 'fn'.")
    return matches


def count_samples_in_match_intervals(match_df: pd.DataFrame, match_type: Literal["tp", "fp", "fn", "tn"]) -> int:
    """Count the number of samples in the intervals of the given match type.

    Parameters
    ----------
    match_df: pd.DataFrame
        A DataFrame containing the categorized intervals with their `start` and `end` index and the respective match
        type in a column named `match_type`.
    match_type: Literal["tp", "fp", "fn", "tn"]
        The match type to count the samples for.
    """
    try:
        _ = match_df[["start", "end", "match_type"]]
    except KeyError as e:
        raise ValueError("`match_df` must have columns named 'start', 'end' and 'match_type'.") from e
    matches = match_df.query(f"match_type == '{match_type}'")
    return count_samples_in_intervals(matches)


def count_samples_in_intervals(intervals: pd.DataFrame) -> int:
    """Count the number of samples in the given interval list."""
    try:
        _ = intervals[["start", "end"]]
    except KeyError as e:
        raise ValueError("`intervals` must have columns named 'start' and 'end'.") from e
    # +1 because the end index is included
    return int((intervals["end"] - intervals["start"] + 1).sum())


def extract_tp_matches(metrics: pd.DataFrame, match_indices: pd.Series) -> pd.DataFrame:
    """
    Extract true positive (TP) matches from the metrics DataFrame based on provided match indices.

    This function is used in :func:`~mobgap.utils.intervals.get_matching_intervals` and
    :func:`~mobgap.initial_contacts.evaluation.get_matching_ics` to filter out the TP matches from
    a larger DataFrame of evaluation metrics (e.g., detected/reference gait sequences or initial
    contacts). TP matches are calculated by :func:`~mobgap.utils.intervals.categorize_intervals`
    and :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list` for gait sequence and initial
    contact detection, respectively.

    Parameters
    ----------
    metrics: pd.DataFrame
        The DataFrame containing metrics from which to extract the matches.
    match_indices: pd.Series
        A Series of indices indicating the rows in `metrics` to be extracted.

    Returns
    -------
    matches: pd.DataFrame
        A DataFrame containing the rows of `metrics` corresponding to the indices in `match_indices`.

    Raises
    ------
    ValueError
        If any of the indices in `match_indices` are not found in the `metrics` DataFrame,
        indicating that there is a mismatch between the provided indices and the data.

    Examples
    --------
    >>> metrics = pd.DataFrame({
    >>>     'step_id': [0, 1, 2, 3],
    >>>     'metric': [597, 660, 720, 797]
    >>> }).set_index("step_id")
    >>> match_indices = pd.Series([0, 2])
    >>> extract_tp_matches(metrics, match_indices)
                metric
    step_id
    0            597
    2            720
    """
    try:
        matches = metrics.loc[match_indices]
    except KeyError as e:
        raise ValueError(
            "The indices from the provided `matches` DataFrame do not fit to the metrics DataFrames. "
            "Please ensure that the `matches` DataFrame is calculated based on the same data "
            "as the `metrics` DataFrames and thus refers to valid indices."
        ) from e
    return matches


def combine_detected_and_reference_metrics(
    detected: pd.DataFrame, reference: pd.DataFrame, tp_matches: Union[pd.DataFrame, None] = None
) -> pd.DataFrame:
    """
    Combine metrics from detected and reference DataFrames using a set of true positive matches to reindex.

    This function is used in :func:`~mobgap.utils.intervals.get_matching_intervals` and
    :func:`~mobgap.initial_contacts.evaluation.get_matching_ics` to merge two DataFrames (`detected`
    and `reference`) based on their common columns. The dataframes are obtained by
    :func:`~mobgap.utils.intervals.extract_tp_matches`. Optionally, if a set of true positive matches
    (`tp_matches`) is provided, it will reindex both DataFrames before merging.
    The result is a combined DataFrame where detected and reference metrics are placed side-by-side
    for each matching index with multi-level columns.

    Parameters
    ----------
    detected : pd.DataFrame
        The DataFrame containing the detected metrics (gait sequences or initial contacts).
    reference : pd.DataFrame
        The DataFrame containing the reference metrics.
    tp_matches : Union[pd.DataFrame, None], optional
        A DataFrame containing true positive matches, by default None. If provided, the indices
        of the `detected` and `reference` DataFrames will be reindexed to match the true positive
        matches before combining. If None, no reindexing is done.

    Returns
    -------
    pd.DataFrame
        A combined DataFrame with the detected and reference metrics side-by-side. The columns
        are multi-indexed with levels corresponding to `detected` and `reference` for easier
        comparison of corresponding metrics.

    Raises
    ------
    ValueError
        If no common columns are found between `detected` and `reference`.

    Notes
    -----
    - The merged DataFrame has multi-level columns, where the top level indicates the source
      (`detected` or `reference`) and the lower level corresponds to the metric name.
    - We add a new column `orig_index` to both DataFrames to keep track of the original index

    """
    common_columns = list(set(reference.columns).intersection(detected.columns))
    if len(common_columns) == 0:
        raise ValueError("No common columns found in `metrics_detected` and `metrics_reference`.")

    detected = detected[common_columns].assign(orig_index=lambda df_: df_.index.to_list()).reset_index(drop=True)
    reference = reference[common_columns].assign(orig_index=lambda df_: df_.index.to_list()).reset_index(drop=True)

    combined = pd.concat({"detected": detected, "reference": reference}, axis=1)
    # make 'metrics' level the uppermost level and sort columns accordingly for readability
    combined = combined.swaplevel(axis=1).sort_index(axis=1, level=0)
    if tp_matches is not None:
        combined.index = tp_matches.index

    return combined


def remove_short_interior_intervals(intervals: np.ndarray, min_samples: int, data_length: int) -> np.ndarray:
    """Remove intervals shorter than ``min_samples`` unless they touch a recording boundary."""
    intervals = np.asarray(intervals, dtype=np.int64)
    if intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    intervals = intervals.reshape(-1, 2)
    if min_samples <= 0:
        return intervals.copy()

    durations = intervals[:, 1] - intervals[:, 0]
    at_boundary = (intervals[:, 0] == 0) | (intervals[:, 1] == data_length)
    return intervals[(durations >= min_samples) | at_boundary]


def find_intersections(intervals_a: np.ndarray, intervals_b: np.ndarray) -> np.ndarray:
    """Find the intersections between two sets of intervals.

    Parameters
    ----------
    intervals_a
        The first list of intervals. Each interval is represented as a tuple of two integers.
    intervals_b
        The second list of intervals. Each interval is represented as a tuple of two integers.

    Returns
    -------
    np.ndarray
        An array of intervals that are the intersections of the intervals in `intervals_a` and `intervals_b`.
        Each interval is represented as a list of two integers.

    """
    # Create Interval Trees
    intervals_a_tree = IntervalTree.from_tuples(intervals_a)
    intervals_b_tree = IntervalTree.from_tuples(intervals_b)

    overlap_intervals = []

    for interval in intervals_b_tree:
        overlaps = sorted(intervals_a_tree.overlap(interval.begin, interval.end))
        if overlaps:
            for overlap in overlaps:
                start = max(interval.begin, overlap.begin)
                end = min(interval.end, overlap.end)
                overlap_intervals.append([start, end])

    return merge_intervals(np.array(overlap_intervals)) if len(overlap_intervals) != 0 else np.array([])


def _complement_intervals(intervals: pd.DataFrame, data_length: int) -> pd.DataFrame:
    if data_length == 0:
        return pd.DataFrame(columns=["start", "end"])

    complement: list[tuple[int, int]] = []
    current_start = 0
    for start, end in intervals[["start", "end"]].itertuples(index=False):
        if start > current_start:
            complement.append((current_start, int(start)))
        current_start = max(current_start, int(end))
    if current_start < data_length:
        complement.append((current_start, data_length))
    return pd.DataFrame(complement, columns=["start", "end"])


def interval_mean(
    measurement_samples: np.ndarray,
    measurements: np.ndarray,
    interval_start_ends: np.ndarray,
) -> np.ndarray:
    """Calculate the average over all measurements within the intervals.

    .. warning :: This assumes that the measurement samples are sorted!

    The method considers all measurements that are within the interval, including the start and end values.

    The calculation is performed using "nanmean", so that NaN values in the measurement are ignored.
    If no measurements are within the interval, the result will be NaN.

    Parameters
    ----------
    measurement_samples
        The samples at which the measurements were taken.
    measurements
        The measurements.
    interval_start_ends
        The start and end indices of the intervals.
        The first column ([:, 0]) contains the start indices and the second column ([:, 1]) contains the end indices.
        The unit of the indices must be the same as the unit of the measurement samples.

    Returns
    -------
    The average over all measurements within the intervals.

    """
    if len(measurement_samples) == 0:
        return np.full(len(interval_start_ends), np.nan)

    if len(interval_start_ends) == 0:
        return np.array([])

    # We use searchsorted to find the start and end indices of the intervals
    interval_start_indices = np.searchsorted(measurement_samples, interval_start_ends[:, 0], side="left")
    interval_end_indices = np.searchsorted(measurement_samples, interval_start_ends[:, 1], side="right")

    # We calculate the average over all measurements within the intervals
    # We suppress the RuntimeWarning for the case that there are no measurements within the interval
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        interval_means = np.array(
            [np.nanmean(measurements[s:e]) for s, e in zip(interval_start_indices, interval_end_indices)]
        )
    return interval_means


def _only_start_end(intervals: pd.DataFrame, *, index_name: Optional[str] = None) -> pd.DataFrame:
    result = intervals[["start", "end"]].astype({"start": "int64", "end": "int64"})
    if index_name is not None:
        return result.rename_axis(index_name)
    return result


__all__ = [
    "bool_array_to_start_end_array",
    "categorize_intervals",
    "categorize_intervals_per_sample",
    "combine_detected_and_reference_metrics",
    "count_samples_in_intervals",
    "count_samples_in_match_intervals",
    "extract_tp_matches",
    "find_intersections",
    "get_matching_intervals",
    "interval_mean",
    "map_intervals_to_regions",
    "merge_intervals",
    "remove_short_interior_intervals",
    "start_end_array_to_bool_array",
]
