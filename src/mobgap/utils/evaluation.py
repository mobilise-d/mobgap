"""Module containing functions to calculate common performance metrics."""

import warnings
from typing import Literal, Union

import pandas as pd

from mobgap.utils._evaluation_challenges import Evaluation, EvaluationCV, save_evaluation_results


def precision_recall_f1_score(
    matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[str, float]:
    """Compute precision, recall and F1-score.

    The precision is the ratio tp / (tp + fp), where tp is the number of true positives and fp the number of false
    positives.
    Intuitively speaking, the precision is the ability of the classifier not to label a negative sample as positive.

    The recall is the ratio tp / (tp + fn), where tp is the number of true positives and fn the number of false
    negatives.
    Intuitively speaking, the recall is the ability of the classifier to find all the positive samples.

    The F1 score can be interpreted as the harmonic mean of precision and recall, where an F1 score reaches its
    best value at 1 and worst score at 0.

    Parameters
    ----------
    matches_df
        A 3 column dataframe.
        Currently supported are dataframes resulting either from the evaluation of gait sequence detection algorithms
        (results from :func:`~mobgap.gait_sequences.evaluation.categorize_intervals_per_sample`),
        or from the evaluation of initial contact detection algorithms
        (results from :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list`).
        To be handled like a gait sequence detection evaluation dataframe,
        the `matches_df` must have columns "start", "end", and "match_type".
        To be handled like an initial contact detection evaluation dataframe,
        the `matches_df` must have columns "ic_id_detected", "ic_id_reference", "match_type".
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives), or "fn" (false negative) or "tn" (true negative).
        Any "tn" matches are ignored in the initial contact match dataframe, since they are not meaningful and normally
        not reported in this application.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    score_metrics : {"precision": precision, "recall": recall, "f1_score": f1_score}
    """
    precisions = precision_score(matches_df, zero_division=zero_division)
    recalls = recall_score(matches_df, zero_division=zero_division)
    f1_scores = f1_score(matches_df, zero_division=zero_division)

    return {
        "precision": precisions,
        "recall": recalls,
        "f1_score": f1_scores,
    }


def precision_score(matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn") -> float:
    """Compute the precision.

    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false
    positives.
    Intuitively speaking, the precision is the ability of the classifier not to label a negative sample as positive.

    The best value is 1 and the worst value is 0.
    Precision can be also referred to as positive predictive value (PPV).

    Parameters
    ----------
    matches_df
        A 3 column dataframe.
        Currently supported are dataframes resulting either from the evaluation of gait sequence detection algorithms
        (results from :func:`~mobgap.gait_sequences.evaluation.categorize_intervals_per_sample`),
        or from the evaluation of initial contact detection algorithms
        (results from :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list`).
        To be handled like a gait sequence detection evaluation dataframe,
        the `matches_df` must have columns "start", "end", and "match_type".
        To be handled like an initial contact detection evaluation dataframe,
        the `matches_df` must have columns "ic_id_detected", "ic_id_reference", "match_type".
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives), or "fn" (false negative) or "tn" (true negative).
        Any "tn" matches are ignored in the initial contact match dataframe, since they are not meaningful and normally
        not reported in this application.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision_score: float
        Value between 0 and 1.
    """
    if _input_is_gsd_matches_df(matches_df):
        tp = count_samples_in_match_intervals(matches_df, "tp")
        fp = count_samples_in_match_intervals(matches_df, "fp")
    elif _input_is_icd_matches_df(matches_df):
        tp = len(matches_df[matches_df["match_type"] == "tp"])
        fp = len(matches_df[matches_df["match_type"] == "fp"])
    else:
        raise ValueError(
            "Only gait sequence detection match dataframes (mandatory columns 'start', "
            "'end', and 'match_type'), or initial contact detection match dataframes "
            "(mandatory columns 'ic_id_detected', 'ic_id_reference', and 'match_type') "
            "are supported as valid input at the moment."
        )

    output = _calculate_score(tp, tp + fp, zero_division=zero_division, caller_function_name="precision")

    return output


def recall_score(matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn") -> float:
    """Compute the recall.

    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false
    negatives.
    Intuitively speaking, the recall is the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.
    Recall can be also referred to as sensitivity, hit rate, or true positive rate (TPR).

    Parameters
    ----------
    matches_df
        A 3 column dataframe.
        Currently supported are dataframes resulting either from the evaluation of gait sequence detection algorithms
        (results from :func:`~mobgap.gait_sequences.evaluation.categorize_intervals_per_sample`),
        or from the evaluation of initial contact detection algorithms
        (results from :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list`).
        To be handled like a gait sequence detection evaluation dataframe,
        the `matches_df` must have columns "start", "end", and "match_type".
        To be handled like an initial contact detection evaluation dataframe,
        the `matches_df` must have columns "ic_id_detected", "ic_id_reference", "match_type".
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives), or "fn" (false negative) or "tn" (true negative).
        Any "tn" matches are ignored in the initial contact match dataframe, since they are not meaningful and normally
        not reported in this application.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    recall_score: float
        Value between 0 and 1.
    """
    if _input_is_gsd_matches_df(matches_df):
        tp = count_samples_in_match_intervals(matches_df, "tp")
        fn = count_samples_in_match_intervals(matches_df, "fn")
    elif _input_is_icd_matches_df(matches_df):
        tp = len(matches_df[matches_df["match_type"] == "tp"])
        fn = len(matches_df[matches_df["match_type"] == "fn"])
    else:
        raise ValueError(
            "Only gait sequence detection match dataframes (mandatory columns 'start', "
            "'end', and 'match_type'), or initial contact detection match dataframes "
            "(mandatory columns 'ic_id_detected', 'ic_id_reference', and 'match_type') "
            "are supported as valid input at the moment."
        )

    output = _calculate_score(tp, tp + fn, zero_division=zero_division, caller_function_name="recall")

    return output


def specificity_score(
    matches_df: pd.DataFrame,
    *,
    n_overall_samples: Union[int, None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
    tn_warning: bool = True,
) -> float:
    """Compute the specificity.

    The specificity is the ratio tn / (tn + fp) where tn is the number of true negatives and fp the number of false
    positives.
    Intuitively speaking, the specificity is the ability of the classifier to find all the negative samples.

    The best value is 1 and the worst value is 0.
    Specificity can be also referred to as the true negative rate (TNR).

    Parameters
    ----------
    matches_df
        A 3 column dataframe.
        Currently supported are dataframes resulting from the evaluation of gait sequence detection
        algorithms (results from :func:`~mobgap.gait_sequences.evaluation.categorize_intervals_per_sample`).
        To be handled like a gait sequence detection evaluation dataframe,
        the `matches_df` must have columns "start", "end", and "match_type".
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives), "fn" (false negative), or "tn" (true negative), if no segmented
        counterpart exists. If the `matches_df` does not contain `tn` matches, the number of overall samples in the
        recording needs to be provided as `n_overall_samples`.
    n_overall_samples: Union[int, None]
        Number of overall samples. Must be provided if the `matches_df` does not contain `tn` matches.
        Needs to be kept as `None` if the `matches_df` contains `tn` matches.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.
    tn_warning : bool, default=True
       A warning is raised if `matches_df` does not contain `tn` matches and `n_overall_samples` is not provided.
       Otherwise, warning is suppressed and tn is set to 0.

    Returns
    -------
    specificity_score: float
        Value between 0 and 1.

    Notes
    -----
    Initial contact detection evaluation dataframes
    (with the columns "ic_id_detected", "ic_id_reference", "match_type") are rejected for this metric as the
    percentage of true negatives outweighs all other match types and thus the specificity score is not meaningful.
    """
    if _input_is_gsd_matches_df(matches_df):
        fp = count_samples_in_match_intervals(matches_df, "fp")
        tn = _estimate_number_tn_samples(matches_df, n_overall_samples, tn_warning=tn_warning)
    elif _input_is_icd_matches_df(matches_df):
        raise ValueError(
            "The sample-wise specificity score is not a meaningful metric for initial contact detection "
            "match dataframes, as the number of true negatives always outweighs the number of "
            "false positives."
        )
    else:
        raise ValueError(
            "Only gait sequence detection match dataframes with the mandatory columns 'start', "
            "'end', and 'match_type' are supported as valid input at the moment."
        )

    output = _calculate_score(tn, tn + fp, zero_division=zero_division, caller_function_name="specificity")

    return output


def accuracy_score(
    matches_df: pd.DataFrame,
    *,
    n_overall_samples: Union[int, None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
    tn_warning: bool = True,
) -> float:
    """Compute the specificity.

    The accuracy is the ratio (tp + tn) / (fp + fn + tp + tn) where tp is the number of true positives,
    tn is the number of true negatives, fp the number of false positives and fn the number of false negatives.
    Intuitively speaking, the accuracy is ratio of correctly detected positives out of all samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    matches_df
        A 3 column dataframe.
        Currently supported are dataframes resulting from the evaluation of gait sequence detection
        algorithms (results from :func:`~mobgap.gait_sequences.evaluation.categorize_intervals_per_sample`).
        To be handled like a gait sequence detection evaluation dataframe,
        the `matches_df` must have columns "start", "end", and "match_type".
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives), "fn" (false negative), or "tn" (true negative), if no segmented
        counterpart exists. If the `matches_df` does not contain `tn` matches, the number of overall samples in the
        recording needs to be provided as `n_overall_samples`.
    n_overall_samples: Union[int, None]
        Number of overall samples. Must be provided if the `matches_df` does not contain `tn` matches.
        Needs to be kept as `None` if the `matches_df` contains `tn` matches.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.
    tn_warning : bool, default=True
       A warning is raised if `matches_df` does not contain `tn` matches and `n_overall_samples` is not provided.
       Otherwise, warning is suppressed and tn is set to 0.

    Returns
    -------
    accuracy_score: float
        Value between 0 and 1.

    Notes
    -----
    Initial contact detection evaluation dataframes
    (with the columns "ic_id_detected", "ic_id_reference", "match_type") are rejected for this metric as the
    percentage of true negatives outweighs all other match types and thus the accuracy score is not meaningful.
    """
    if _input_is_gsd_matches_df(matches_df):
        tp = count_samples_in_match_intervals(matches_df, "tp")
        fn = count_samples_in_match_intervals(matches_df, "fn")
        fp = count_samples_in_match_intervals(matches_df, "fp")
        tn = _estimate_number_tn_samples(matches_df, n_overall_samples, tn_warning=tn_warning)
    elif _input_is_icd_matches_df(matches_df):
        raise ValueError(
            "The sample-wise accuracy score is not a meaningful metric for initial contact detection match dataframes, "
            "as the number of true negatives always outweighs the number of false negatives, "
            "false positives, and true positives."
        )
    else:
        raise ValueError(
            "Only gait sequence detection match dataframes with the mandatory columns 'start', "
            "'end', and 'match_type' are supported as valid input at the moment."
        )

    output = _calculate_score(tn + tp, fp + fn + tp + tn, zero_division=zero_division, caller_function_name="accuracy")

    return output


def npv_score(
    matches_df: pd.DataFrame,
    *,
    n_overall_samples: Union[int, None] = None,
    zero_division: Literal["warn", 0, 1] = "warn",
    tn_warning: bool = True,
) -> float:
    """Compute the negative predictive value (NPV).

    The NPV is the ratio tn / (tn + fn) where tn is the number of true negatives and fn the number of false
    negatives.
    Intuitively speaking, the NPV is the ability of the classifier not to label a positive sample as negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    matches_df
        A 3 column dataframe.
        Currently supported are dataframes resulting from the evaluation of gait sequence detection
        algorithms (results from :func:`~mobgap.gait_sequences.evaluation.categorize_intervals_per_sample`).
        To be handled like a gait sequence detection evaluation dataframe,
        the `matches_df` must have columns "start", "end", and "match_type".
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives), "fn" (false negative), or "tn" (true negative), if no segmented
        counterpart exists. If the `matches_df` does not contain `tn` matches, the number of overall samples in the
        recording needs to be provided as `n_overall_samples`.
    n_overall_samples: Union[int, None]
        Number of overall samples. Must be provided if the `matches_df` does not contain `tn` matches.
        Needs to be kept as `None` if the `matches_df` contains `tn` matches.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.
    tn_warning : bool, default=True
        A warning is raised if `matches_df` does not contain `tn` matches and `n_overall_samples` is not provided.
        Otherwise, warning is suppressed and tn is set to 0.

    Returns
    -------
    npv_score: float
        Value between 0 and 1.

    Notes
    -----
    Initial contact detection evaluation dataframes
    (with the columns "ic_id_detected", "ic_id_reference", "match_type") are rejected for this metric as the
    percentage of true negatives outweighs all other match types and thus the NPV score is not meaningful.
    """
    if _input_is_gsd_matches_df(matches_df):
        fn = count_samples_in_match_intervals(matches_df, "fn")
        tn = _estimate_number_tn_samples(matches_df, n_overall_samples, tn_warning=tn_warning)
    elif _input_is_icd_matches_df(matches_df):
        raise ValueError(
            "The sample-wise NPV score is not a meaningful metric for initial contact detection match dataframes, as "
            "the number of true negatives always outweighs the number of false negatives."
        )
    else:
        raise ValueError(
            "Only gait sequence detection match dataframes with the mandatory columns 'start', "
            "'end', and 'match_type' are supported as valid input at the moment."
        )

    output = _calculate_score(tn, tn + fn, zero_division=zero_division, caller_function_name="npv")

    return output


def f1_score(matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn") -> float:
    """Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as the harmonic mean of precision and recall, where an F1 score reaches its
    best value at 1 and worst score at 0.

    The formula for the F1 score is:
    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    matches_df
        A 3 column dataframe.
        Currently supported are dataframes resulting either from the evaluation of gait sequence detection algorithms
        (results from :func:`~mobgap.gait_sequences.evaluation.categorize_intervals_per_sample`),
        or from the evaluation of initial contact detection algorithms
        (results from :func:`~mobgap.initial_contacts.evaluation.categorize_ic_list`).
        To be handled like a gait sequence detection evaluation dataframe,
        the `matches_df` must have columns "start", "end", and "match_type".
        To be handled like an initial contact detection evaluation dataframe,
        the `matches_df` must have columns "ic_id_detected", "ic_id_reference", "match_type".
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives), or "fn" (false negative) or "tn" (true negative).
        Any "tn" matches are ignored in the initial contact match dataframe, since they are not meaningful and normally
        not reported in this application.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    f1_score: float
        Value between 0 and 1.
    """
    recall = recall_score(matches_df.copy())
    precision = precision_score(matches_df.copy())
    output = _calculate_score(
        2 * (precision * recall),
        precision + recall,
        zero_division=zero_division,
        caller_function_name="f1",
    )

    return output


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

    This function is used in :func:`~mobgap.gait_sequences.evaluation.get_matching_intervals` and
    :func:`~mobgap.initial_contacts.evaluation.get_matching_ics` to filter out the TP matches from
    a larger DataFrame of evaluation metrics (e.g., detected/reference gait sequences or initial
    contacts). TP matches are calculated by :func:`~mobgap.gait_sequences.evaluation.categorize_intervals`
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

    This function is used in :func:`~mobgap.gait_sequences.evaluation.get_matching_intervals` and
    :func:`~mobgap.initial_contacts.evaluation.get_matching_ics` to merge two DataFrames (`detected`
    and `reference`) based on their common columns. The dataframes are obtained by
    :func:`~mobgap.utils.evaluation.extract_tp_matches`. Optionally, if a set of true positive matches
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


def _estimate_number_tn_samples(matches_df: pd.DataFrame, n_overall_samples: Union[int, None], tn_warning: bool) -> int:
    tn = count_samples_in_match_intervals(matches_df, "tn")
    if tn > 0 and n_overall_samples is not None:
        raise ValueError("If `matches_df` contains `tn` matches `n_overall_samples` must not be provided.")
    if tn == 0 and n_overall_samples is None and tn_warning:
        warnings.warn(
            "`matches_df` does not contain `tn` matches. Thus, number of true negative samples is "
            "assumed to be 0. If this is not the case, please provide `n_overall_samples`, so that the "
            "number of true negatives can be derived.",
            stacklevel=2,
        )
    if tn == 0 and n_overall_samples is not None:
        tn = (
            n_overall_samples
            - count_samples_in_match_intervals(matches_df, "fp")
            - count_samples_in_match_intervals(matches_df, "tp")
            - count_samples_in_match_intervals(matches_df, "fn")
        )
    return tn


def _calculate_score(a: float, b: float, *, zero_division: Literal["warn", 0, 1], caller_function_name: str) -> float:
    try:
        return a / b
    except ZeroDivisionError as e:
        if zero_division == "warn":
            warnings.warn(
                f"Zero division happened while calculating the {caller_function_name} score. Returning 0",
                UserWarning,
                stacklevel=2,
            )
            return 0

        if zero_division in [0, 1]:
            return zero_division

        raise ValueError('"zero_division" must be set to "warn", 0 or 1!') from e


def _input_is_gsd_matches_df(matches_df: pd.DataFrame) -> bool:
    try:
        _ = matches_df[["start", "end", "match_type"]]
    except KeyError:
        return False
    return True


def _input_is_icd_matches_df(matches_df: pd.DataFrame) -> bool:
    try:
        _ = matches_df[["ic_id_detected", "ic_id_reference", "match_type"]]
    except KeyError:
        return False
    return True


__all__ = [
    "Evaluation",
    "EvaluationCV",
    "accuracy_score",
    "combine_detected_and_reference_metrics",
    "count_samples_in_intervals",
    "count_samples_in_match_intervals",
    "extract_tp_matches",
    "f1_score",
    "npv_score",
    "precision_recall_f1_score",
    "precision_score",
    "recall_score",
    "save_evaluation_results",
    "specificity_score",
]
