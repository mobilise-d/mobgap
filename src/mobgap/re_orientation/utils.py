"""Select recorded corrections for sequences in a shared coordinate system."""

from typing import TypeVar

import numpy as np
import pandas as pd

from mobgap.utils.intervals import map_intervals_to_regions

CorrectionT = TypeVar("CorrectionT")


def get_orientation_for_sequences(
    sequences: pd.DataFrame, reorientation_sections: pd.DataFrame, *, default: CorrectionT
) -> pd.Series:
    """Select an opaque correction for each sequence without extrapolating it.

    Parameters
    ----------
    sequences
        Dataframe with ``start`` and ``end`` columns. The same table can describe
        walking bouts, turns, or other intervals. See
        :func:`~mobgap.utils.intervals.map_intervals_to_regions` for coordinate and
        index requirements. No parent gait-sequence identifier is needed.
    reorientation_sections
        Dataframe with ``start``, ``end``, ``orientation``, and ``correction_applied``
        columns. ``orientation`` stores a method-specific correction value.
        ``correction_applied`` explicitly marks sections requiring correction.
        A sequence fully contained in such a section receives its recorded value.
    default
        The caller's nominal or identity correction, used when a sequence does not
        overlap a section requiring correction. This value must be suitable for
        the reorientation method's ``apply_correction`` operation.

    Returns
    -------
    pandas.Series
        Object Series named ``orientation``, aligned exactly to ``sequences.index``.
        Correction objects and the default are preserved unchanged, without copying.

    Raises
    ------
    ValueError
        If a sequence has a positive-length partial overlap with any section
        requiring correction, or multiple sections fully contain it. Touching
        half-open boundaries do not overlap. Partial overlaps with nominal sections
        are allowed.

    Notes
    -----
    This helper does not interpret correction values or infer identity from them.
    A recorded correction can represent an unresolved detection fallback. Selecting
    or applying it does not prove that orientation was resolved.

    Examples
    --------
    Prepare the table once before iterating over signal slices:

    >>> wbs = wbs.assign(
    ...     orientation=lambda rows: get_orientation_for_sequences(
    ...         rows, sections, default=identity_correction
    ...     )
    ... )
    """
    required = reorientation_sections.loc[reorientation_sections["correction_applied"]]
    required = required.loc[required["start"] < required["end"]]
    # A positive-length partial overlap puts at least one section boundary strictly
    # inside the query. Sorted boundary searches also detect entering a section
    # from outside, and crossing several sections, without an N-by-M matrix.
    for column in ("start", "end"):
        boundaries = np.sort(required[column].to_numpy())
        left = np.searchsorted(boundaries, sequences["start"].to_numpy(), side="right")
        right = np.searchsorted(boundaries, sequences["end"].to_numpy(), side="left")
        if np.any(right > left):
            raise ValueError("A sequence partially overlaps a section requiring orientation correction.")

    matches = map_intervals_to_regions(sequences, reorientation_sections)
    values = []
    # Scalar access keeps tuple identifiers opaque, including single-level tuple indexes.
    for section_id in matches:
        if section_id is pd.NA or not reorientation_sections.at[section_id, "correction_applied"]:  # noqa: PD008
            values.append(default)
        else:
            values.append(reorientation_sections.at[section_id, "orientation"])  # noqa: PD008
    return pd.Series(values, index=sequences.index, name="orientation", dtype=object)


__all__ = ["get_orientation_for_sequences"]
