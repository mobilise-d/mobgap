from typing import Union, Tuple

import numpy as np
import pandas as pd

from scipy.spatial import KDTree


def evaluate_initial_contact_list(
    *,
    reference_ic_list: pd.DataFrame,
    detected_ic_list: pd.DataFrame,
    tolerance: Union[int, float] = 0, # TODO: insert practical default value
    detected_ic_list_postfix: str = "_detected",
    reference_ic_list_postfix: str = "_reference",
) -> pd.DataFrame:
    f"""Find True Positives, False Positives and True Negatives by comparing an initial contact list with a reference.

    This compares an initial contact event list with a ground truth initial contact list and returns true positive,
    false positive and true negative matches.
    The comparison is purely based on the index of each initial contact in the lists.
    Two initial contacts are considered a positive match, if the difference between their indices is less than or equal
    to the threshold.

    If multiple initial contacts of the initial contact list would match to a single ground truth initial contact
    (or vise-versa), only the initial contact with the lowest distance is considered an actual match.
    In case of multiple matches with the same distance, the first match will be considered.

    Parameters
    ----------
    reference_ic_list
        The ground truth initial contact list.
    detected_ic_list
        The list of detected initial contacts.
    tolerance
        The allowed tolerance between labels in samples.
        The comparison is done as `distance <= tolerance`.
    detected_ic_list_postfix
        A postfix that will be appended to the index name of the initial contact list in the output.
    reference_ic_list_postfix
        A postfix that will be appended to the index name of the reference list in the output.

    Returns
    -------
    matches
        A 3 column dataframe with the column names `s_id{detected_ic_list_postfix}`, `s_id{reference_ic_list_postfix}` and
        `match_type`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        The `match_type` column indicates the type of match.
        For all initial contacts that have a match in the ground truth list, this will be "tp" (true positive).
        Initial contacts that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positives)
        All reference initial contacts that do not have a counterpart in the detected list 
        are marked as "fn" (false negative).

    Examples
    --------
    >>> ic_list_reference = pd.DataFrame([10, 20, 32, 40], columns=["ic"]).rename_axis("s_id")
    >>> ic_list_detected = pd.DataFrame([11, 23, 30, 50], columns=["ic"]).rename_axis("s_id")
    >>> matches = evaluate_initial_contact_list(
    ...     reference_ic_list=ic_list_reference, detected_ic_list=ic_list_detected, tolerance=2
    ... )
    >>> matches
      ic_id_detected ic_id_reference match_type
    0    0                 0         tp
    1    1               NaN         fp
    2    2                 2         tp
    3    3               NaN         fp
    4  NaN                 1         fn
    5  NaN                 3         fn
    """
    # TODO: add input validation here

    if detected_ic_list_postfix == reference_ic_list_postfix:
        raise ValueError("The postfix for the left and the right initial contact list must be different.")

    if tolerance < 0:
        raise ValueError("The tolerance must be larger 0.")

    match_cols = ["ic"]

    if not (
        set(match_cols).issubset(detected_ic_list.columns)
        and set(match_cols).issubset(reference_ic_list.columns)
    ):
        raise ValueError(
            f"One or more selected columns ({match_cols}) are missing in at least one of the provided "
            f"initial contact lists"
        )
    # TODO: set correct index?

    left_indices, right_indices = _match_label_lists(
        detected_ic_list[match_cols].to_numpy(),
        reference_ic_list[match_cols].to_numpy(),
        tolerance=tolerance,
    )

    detected_index_name = detected_ic_list.index.name + detected_ic_list_postfix
    reference_index_name = reference_ic_list.index.name + reference_ic_list_postfix

    matches_detected = pd.DataFrame(index=detected_ic_list.index.copy(), columns=[reference_index_name])
    matches_detected.index.name = detected_index_name

    matches_reference = pd.DataFrame(index=reference_ic_list.index.copy(), columns=[detected_index_name])
    matches_reference.index.name = reference_index_name

    ic_list_detected_idx = detected_ic_list.iloc[left_indices].index
    ic_list_reference_idx = reference_ic_list.iloc[right_indices].index

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

    segmented_index_name = detected_ic_list.index.name + detected_ic_list_postfix
    ground_truth_index_name = reference_ic_list.index.name + reference_ic_list_postfix
    matches.loc[~matches.isna().any(axis=1), "match_type"] = "tp"
    matches.loc[matches[ground_truth_index_name].isna(), "match_type"] = "fp"
    matches.loc[matches[segmented_index_name].isna(), "match_type"] = "fn"

    return matches


def _match_label_lists(
    list_left: np.ndarray, list_right: np.ndarray, tolerance: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
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


if __name__ == '__main__':
    ic_list_reference = pd.DataFrame([10, 20, 32, 40], columns=["ic"]).rename_axis("s_id")
    ic_list_detected = pd.DataFrame([11, 23, 30, 50], columns=["ic"]).rename_axis("s_id")
    matches = evaluate_initial_contact_list(
        reference_ic_list=ic_list_reference, detected_ic_list=ic_list_detected, tolerance=2
    )
    print(matches)
