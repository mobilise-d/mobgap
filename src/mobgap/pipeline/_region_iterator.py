"""Collect Series results from signal regions into indexed DataFrames."""

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

import pandas as pd
from tpcp.misc import BaseTypedIterator

from mobgap.pipeline._gs_iterator import Region, RegionDataTuple, iter_gs

ResultT = TypeVar("ResultT")


class RegionIterator(BaseTypedIterator[RegionDataTuple, ResultT], Generic[ResultT]):
    """Iterate signal regions and assemble each Series result field into a DataFrame.

    Each iteration yields ``((region, window), result)`` using the existing :class:`Region` type. Read additional
    metadata directly from the source table, for example ``regions.loc[region.id, "label"]``. The iterator only
    slices data and assembles results; any signal preparation or coordinate rebasing belongs in the loop body.

    Define a result dataclass with Series fields and pass it as ``data_type``. Assign one Series to each field in
    the loop. After iteration, the same field in ``results_`` contains a DataFrame whose columns are the Series
    labels and whose rows preserve the source region index and order, including all MultiIndex levels. Series names
    are ignored, different Series labels are aligned as columns, and unset results are omitted independently for each
    field. MultiIndex Series labels become MultiIndex columns with the same level names.
    There are no coordinate offsets or statistical reductions. An empty iteration produces empty DataFrames
    with the source index schema and no columns.

    The iterator retains window data and individual results in ``raw_results_``. Reusing it starts a fresh iteration.

    Parameters
    ----------
    data_type
        A caller-defined result dataclass. Every field is assembled from Series into a DataFrame by default.
        As with :class:`~tpcp.misc.BaseTypedIterator`, field types differ between individual and assembled results.
    aggregations
        Optional custom assembly functions, as in :class:`GsIterator`. These override the default for named fields
        and can also assemble other result types.
    """

    def iterate(self, data: pd.DataFrame, region_list: pd.DataFrame) -> Iterator[tuple[RegionDataTuple, ResultT]]:
        """Yield half-open positional windows and empty typed results.

        Parameters
        ----------
        data
            Signal data. Slicing uses sample positions and preserves the original data index.
        region_list
            Table with integer ``start``/``end`` sample positions relative to ``data`` and unique row identifiers.
            Its Index or MultiIndex provides the complete region identity. Rows can overlap or be unsorted.
            Other columns remain in this table and are not copied into ``Region`` or the results.

        Yields
        ------
        region_data
            Existing ``Region`` and the signal slice. ``region.id`` is the original index label or composite tuple.
            For a MultiIndex, ``region.id_origin`` is ``None``; assembled results retain all index-level names.
        result
            An empty instance of ``data_type`` to fill inside the loop.
        """
        self._region_index = region_list.index.copy()
        numbered = region_list[["start", "end"]].reset_index(drop=True).rename_axis("region_position")
        windows = (
            RegionDataTuple(
                Region(self._region_index[region.id], region.start, region.end, region_list.index.name), window
            )
            for region, window in iter_gs(data, numbered, id_col="region_position")
        )
        yield from self._iterate(windows)

    def _get_default_agg(self, field_name: str) -> Callable:
        def assemble_series(values: list[BaseTypedIterator.IteratorResult]) -> pd.DataFrame:
            rows = []
            positions = []
            for position, value in enumerate(values):
                row = getattr(value.result, field_name)
                if row is self.NULL_VALUE:
                    continue
                rows.append(row)
                positions.append(position)
            result = pd.DataFrame(rows)
            result.index = self._region_index.take(positions)
            return result

        return assemble_series
