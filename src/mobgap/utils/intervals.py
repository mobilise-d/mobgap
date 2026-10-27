"""Map intervals that share a coordinate system."""

from heapq import heappush, heappushpop

import pandas as pd


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


__all__ = ["map_intervals_to_regions"]
