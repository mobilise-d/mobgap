"""Class to evaluate initial contact detection algorithms."""
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from gaitlink.utils.evaluation import precision_recall_f1_score


def calculate_icd_performance_metrics(
    ic_list_detected: pd.DataFrame,
    ic_list_reference: pd.DataFrame,
    tolerance: Union[int, float] = 0,
) -> dict[str, Union[float, int]]:
    """
    Calculate performance metrics for initial contact detection results.

    The detected and reference initial contact dataframes must have a column named "ic" that contains the index of the
    resective initial contact.

    The following metrics are calculated:

    - tp_samples`: Number of samples that are correctly detected as initial contacts.
    - `fp_samples`: Number of samples that are falsely detected as initial contacts.
    - `fn_samples`: Number of samples that are not detected as initial contacts.
    - `precision`: Precision of the detected initial contacts.
    - `recall`: Recall of the detected initial contacts.
    - `f1_score`: F1 score of the detected initial contacts.

    See the documentation of :func:`~gaitlink.utils.evaluation.precision_recall_f1_score` for more details about recall,
    precision, and F1 score.

    Parameters
    ----------
    ic_list_detected: pd.DataFrame
        The dataframe of detected initial contacts.
    ic_list_reference: pd.DataFrame
        The ground truth initial contact dataframe.
    tolerance: int or float, optional
        The allowed tolerance between a detected and reference initial contact in samples for it to be considered a
        true positive match.
        The comparison is done as `distance <= tolerance`.

    Returns
    -------
    icd_metrics: dict
    """
    evaluated_matches = evaluate_initial_contact_list(ic_list_detected, ic_list_reference, tolerance=tolerance)

    # estimate tp, fp, fn
    tp_samples = len(evaluated_matches[evaluated_matches["match_type"] == "tp"])
    fp_samples = len(evaluated_matches[evaluated_matches["match_type"] == "fp"])
    fn_samples = len(evaluated_matches[evaluated_matches["match_type"] == "fn"])

    # estimate performance metrics
    precision_recall_f1 = precision_recall_f1_score(evaluated_matches)

    icd_metrics = {
        "tp_samples": tp_samples,
        "fp_samples": fp_samples,
        "fn_samples": fn_samples,
        **precision_recall_f1,
    }

    return icd_metrics


def evaluate_initial_contact_list(
    ic_list_detected: pd.DataFrame,
    ic_list_reference: pd.DataFrame,
    *,
    tolerance: Union[int, float] = 0,  # TODO: insert practical default value?
) -> pd.DataFrame:
    """Evaluate an initial contact list against a reference contact-by-contact.

    This compares an initial contact dataframe with a ground truth initial contact dataframe and returns true positive,
    false positive and true negative matches.
    The comparison is purely based on the index of each initial contact.
    Two initial contacts are considered a positive match, if the difference between their indices is less than or equal
    to the threshold.

    The detected and reference initial contact dataframes must have a column named "ic" that contains the index of the
    resective initial contact.
    As index, a range index or a column of unique identifiers for each initial contact can be provided.

    If multiple detected initial contacts would match to a single ground truth initial contact
    (or vise-versa), only the initial contact with the lowest distance is considered an actual match.
    In case of multiple matches with the same distance, the first match will be considered.

    Parameters
    ----------
    ic_list_detected
        The dataframe of detected initial contacts.
    ic_list_reference
        The ground truth initial contact dataframe.
    tolerance
        The allowed tolerance between a detected and reference initial contact in samples for it to be considered a
        true positive match.
        The comparison is done as `distance <= tolerance`.

    Returns
    -------
    matches
        A 3 column dataframe with the column names `s_id_detected`, `s_id_reference`
        and `match_type`.
        Each row is a match containing the index value of the detected and the reference list, that belong together.
        The `match_type` column indicates the type of match.
        For all initial contacts that have a match in the reference list, this will be "tp" (true positive).
        Initial contacts that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positives)
        All reference initial contacts that do not have a counterpart in the detected list
        are marked as "fn" (false negative).

    Examples
    --------
    >>> ic_detected = pd.DataFrame([11, 23, 30, 50], columns=["ic"]).rename_axis("s_id")
    >>> ic_reference = pd.DataFrame([10, 20, 32, 40], columns=["ic"]).rename_axis("s_id")
    >>> result = evaluate_initial_contact_list(
    ...     ic_list_detected=ic_detected, ic_list_reference=ic_reference, tolerance=2
    ... )
    >>> result
      ic_id_detected ic_id_reference match_type
    0    0                 0         tp
    1    1               NaN         fp
    2    2                 2         tp
    3    3               NaN         fp
    4  NaN                 1         fn
    5  NaN                 3         fn
    """
    detected, reference = _check_input_sanity(ic_list_detected, ic_list_reference)

    if tolerance < 0:
        raise ValueError("The tolerance must be larger 0.")

    left_indices, right_indices = _match_label_lists(
        detected.to_numpy(),
        reference.to_numpy(),
        tolerance=tolerance,
    )

    detected_index_name = ic_list_detected.index.name + "_detected"
    reference_index_name = ic_list_reference.index.name + "_reference"

    matches_detected = pd.DataFrame(index=ic_list_detected.index.copy(), columns=[reference_index_name])
    matches_detected.index.name = detected_index_name

    matches_reference = pd.DataFrame(index=ic_list_reference.index.copy(), columns=[detected_index_name])
    matches_reference.index.name = reference_index_name

    ic_list_detected_idx = ic_list_detected.iloc[left_indices].index
    ic_list_reference_idx = ic_list_reference.iloc[right_indices].index

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

    matches.loc[~matches.isna().any(axis=1), "match_type"] = "tp"
    matches.loc[matches[reference_index_name].isna(), "match_type"] = "fp"
    matches.loc[matches[detected_index_name].isna(), "match_type"] = "fn"

    return matches


def _match_label_lists(
    list_left: np.ndarray, list_right: np.ndarray, tolerance: Union[int, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Find matches in two lists based on the distance between their vectors.

    Parameters
    ----------
    list_left : array with shape (n, 1)
        An n long array of 1-dimensional vectors
    list_right : array with shape (m, 1)
        An m long array of 1-dimensional vectors
    tolerance
        Max allowed Chebyshev distance between matches.
        The comparison is done as "distance <= tolerance".

    Returns
    -------
    left_indices
        Indices from the left list that have a match in the right list.
    right_indices
        Indices from the right list that have a match in the left list.
        A valid match pair is then `(left_indices[i], right_indices[i]) for all i.

    Notes
    -----
    Only a single match per index is allowed in both directions.
    This means that every index will only occur once in the output arrays.
    If an index in one list does have two equally close matches in the other list,
    only the first match will be returned.
    """
    if len(list_left) == 0 or len(list_right) == 0:
        return np.array([]), np.array([])

    right_tree = KDTree(list_right)
    left_tree = KDTree(list_left)

    # p = 1 is used to select the Manhattan distance
    l_nearest_distance, l_nearest_neighbor = right_tree.query(list_left, workers=-1)
    _, r_nearest_neighbor = left_tree.query(list_right, workers=-1)

    # Filter the once that are true one-to-one matches
    l_indices = np.arange(len(list_left))
    combined_indices = np.vstack([l_indices, l_nearest_neighbor]).T
    boolean_map = r_nearest_neighbor[l_nearest_neighbor] == l_indices
    valid_matches = combined_indices[boolean_map]

    # Only keep the matches that are within the tolerance
    valid_matches_distance = l_nearest_distance[boolean_map]
    valid_matches_idx = np.where(valid_matches_distance <= tolerance)[0]
    valid_matches = valid_matches[valid_matches_idx]

    valid_matches = valid_matches.T
    return valid_matches[0], valid_matches[1]


def _check_input_sanity(
    ic_list_detected: pd.DataFrame, ic_list_reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # check if inputs are dataframes
    if not isinstance(ic_list_detected, pd.DataFrame) or not isinstance(ic_list_reference, pd.DataFrame):
        raise TypeError("`ic_list_detected` and `ic_list_reference` must be of type `pandas.DataFrame`.")
    # check if `ic` column is present in both dataframes
    try:
        detected, reference = ic_list_detected[["ic"]], ic_list_reference[["ic"]]
    except KeyError as e:
        raise ValueError("Both `ic_list_detected` and `ic_list_reference` must have a column named `ic`.") from e
    # check if indices are unique
    if not ic_list_detected.index.is_unique or not ic_list_reference.index.is_unique:
        raise ValueError("The indices of both `ic_list_detected` and `ic_list_reference` must be unique.")
    # if index is a range index, rename it to "ic_id"
    if ic_list_detected.index.name is None:
        ic_list_detected.index.name = "ic_id"
    if ic_list_reference.index.name is None:
        ic_list_reference.index.name = "ic_id"
    return detected, reference


if __name__ == "__main__":
    ic_list_detected = pd.DataFrame([11, 23, 30, 50], columns=["ic"]).rename_axis("ic_id")
    ic_list_reference = pd.DataFrame([10, 20, 32, 40], columns=["ic"]).rename_axis("ic_id")
    matches = evaluate_initial_contact_list(
        ic_list_reference=ic_list_reference, ic_list_detected=ic_list_detected, tolerance=2
    )
    print(matches)
