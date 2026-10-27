"""Collect one row of secondary outcomes per signal window."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from tpcp import cf
from tpcp.misc import BaseTypedIterator

from mobgap.pipeline._gs_iterator import Region, RegionDataTuple, iter_gs


@dataclass
class SdmoResult:
    """Typed result for :class:`SdmoIterator`.

    Attributes
    ----------
    parameters
        A one-row DataFrame of outcomes for the current window. The row's index is ignored during assembly.
        After iteration, this field contains the combined table indexed by the original sequence identifiers.
    """

    parameters: pd.DataFrame


class SdmoIterator(BaseTypedIterator[RegionDataTuple, SdmoResult]):
    """Iterate WB or turn windows and assemble one outcome row per sequence.

    Each iteration yields ``((region, window), result)`` using the existing :class:`Region` type. Read additional
    metadata directly from the source table, for example ``sequences.loc[region.id, "orientation"]``. The iterator
    only slices data and assembles results; correction application, stride selection and coordinate rebasing belong
    in the loop body. Signal-based outcomes require correctly oriented body-frame data.

    Set ``result.parameters`` to a one-row DataFrame. The combined ``results_.parameters`` preserves the source
    table's index and row order, including all levels of a turn MultiIndex. It adds no inner row-index level and
    performs no statistical aggregation or coordinate offsets. Leaving a result unset omits that sequence.
    An empty iteration returns an empty table with the source index schema and no outcome columns.

    This iterator retains window data and individual results in ``raw_results_``. Reusing it starts a fresh iteration.
    Use separate instances when both WB and turn results need to remain available.

    Parameters
    ----------
    data_type
        Result dataclass. The default :class:`SdmoResult` provides the ``parameters`` DataFrame field.
    aggregations
        Optional custom result assembly functions, as in :class:`GsIterator`. By default, ``parameters`` is assembled
        into one table and any additional fields in a custom dataclass are collected as lists.
    """

    def __init__(
        self,
        data_type: type[SdmoResult] = SdmoResult,
        aggregations: Sequence[tuple[str, Callable[[list[BaseTypedIterator.IteratorResult]], Any]]] = cf([]),
    ) -> None:
        super().__init__(data_type, aggregations)

    def iterate(self, data: pd.DataFrame, sequences: pd.DataFrame) -> Iterator[tuple[RegionDataTuple, SdmoResult]]:
        """Yield half-open positional windows and empty typed results.

        Parameters
        ----------
        data
            Recording signal. Slicing uses sample positions and preserves the original data index.
        sequences
            Table with integer ``start``/``end`` sample positions relative to ``data`` and unique row identifiers.
            Its Index or MultiIndex provides the complete sequence identity. Rows can overlap or be unsorted.
            Other columns remain in this table and are not copied into ``Region`` or the outcomes.

        Yields
        ------
        region_data
            Existing ``Region`` and the signal slice. ``region.id`` is the original index label or composite tuple.
            For a MultiIndex, ``region.id_origin`` is ``None``; the assembled result retains all index-level names.
        result
            An empty :class:`SdmoResult` to fill inside the loop.
        """
        self._sequence_index = sequences.index.copy()
        numbered = sequences[["start", "end"]].reset_index(drop=True).rename_axis("sequence_position")
        windows = (
            RegionDataTuple(
                Region(self._sequence_index[region.id], region.start, region.end, sequences.index.name), window
            )
            for region, window in iter_gs(data, numbered, id_col="sequence_position")
        )
        yield from self._iterate(windows)

    def _get_default_agg(self, field_name: str) -> Callable:
        if field_name == "parameters":
            return self._assemble_parameters
        return super()._get_default_agg(field_name)

    def _assemble_parameters(self, values: list[BaseTypedIterator.IteratorResult]) -> pd.DataFrame:
        frames = []
        positions = []
        for position, value in enumerate(values):
            frame = value.result.parameters
            if frame is self.NULL_VALUE:
                continue
            if len(frame) != 1:
                raise ValueError("Each SDMO result must contain exactly one row of parameters.")
            frames.append(frame)
            positions.append(position)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result.index = self._sequence_index.take(positions)
        return result
