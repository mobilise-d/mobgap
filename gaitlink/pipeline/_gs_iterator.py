from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Generic, TypeVar

import pandas as pd
from tpcp import cf
from tpcp.misc import BaseTypedIterator
from typing_extensions import TypeAlias

DataclassT = TypeVar("DataclassT")


def iter_gs(data: pd.DataFrame, gs_list: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Iterate over the data based on the given gait-sequences.

    This will yield a dataframe for each gait-sequence.
    We assume that the gait-sequences are sorted by start time and contain a ``start`` and ``end`` column that match
    the units of the data index.

    Parameters
    ----------
    data
        The data to iterate over.
    gs_list
        The list of gait-sequences.

    Yields
    ------
    str
       The gait-sequence-id (i.e. the index from the gs-dataframe).
    pd.DataFrame
        The data of a single gait-sequence.
        Note, that we don't change the index of the data.
        If the data was using an index that started at the beginning of the recording, the index of the individual
        sequences will still be relative to the beginning of the recording.

    """
    for gs in gs_list.reset_index().itertuples(index=False, name="gs"):
        yield gs, data.iloc[gs.start : gs.end]


@dataclass
class FullPipelinePerGsResult:
    initial_contacts: pd.DataFrame
    cadence: pd.Series
    stride_length: pd.Series
    gait_speed: pd.Series


_inputs_type: TypeAlias = tuple[tuple, pd.DataFrame]

T = TypeVar("T")
_aggregator_type: TypeAlias = Callable[[list[_inputs_type], list[pd.DataFrame]], T]


def create_aggregate_df(fix_gs_offset_cols: Sequence[str] = ("start", "end")) -> _aggregator_type[pd.DataFrame]:
    def aggregate_df(inputs: list[_inputs_type], outputs: list[pd.DataFrame]) -> pd.DataFrame:
        sequences, _ = zip(*inputs)
        iter_index_name = sequences[0]._fields[0]

        to_concat = {}
        for gs, o in zip(sequences, outputs):
            if not isinstance(o, pd.DataFrame):
                raise TypeError(f"Expected dataframe for this aggregator, but got {type(o)}")
            if fix_gs_offset_cols:
                cols_to_fix = set(fix_gs_offset_cols).intersection(o.columns)
                o[list(cols_to_fix)] += gs.start
            to_concat[gs[0]] = o

        return pd.concat(to_concat, names=[iter_index_name, *outputs[0].index.names])

    return aggregate_df


@dataclass
class Aggregators:
    """Available aggregators for the gait-sequence iterator.

    Note, that all of them are constructors for aggregators, as they have some configuration options.
    To use them as aggregators, you need to call them with the desired configuration.

    Examples
    --------
    >>> from gaitlink.pipeline import GsIterator
    >>> my_aggregation = [("my_result", GsIterator.DEFAULT_AGGREGATORS.create_aggregate_df(["my_col"]))]
    >>> iterator = GsIterator(aggregations=my_aggregation)

    """

    create_aggregate_df: Callable[[list, list], pd.DataFrame] = create_aggregate_df


class GsIterator(BaseTypedIterator, Generic[DataclassT]):
    """Iterator to split data into gait-sequences and iterate over them individually.

    This can be used to easily iterate over gait-sequences and apply algorithms to them, and then collect the results
    in a convenient way.

    Note that you need to specify the expected results by creating a custom dataclass (learn more in the example linked
    at the bottom of this page).
    Each result can further be aggregated by providing an aggregation function.

    Parameters
    ----------
    data_type
        A dataclass that defines the result type you expect from each iteration.
    aggregations
        An optional list of aggregations to apply to the results.
        This has the form ``[(result_name, aggregation_function), ...]``.
        If a result-name is in the list, the aggregation will be applied to it, when accessing the respective result
        attribute (i.e. ``{result_name}_``).
        If no aggregation is defined for a result, a simple list of all results will be returned.
    NULL_VALUE
        (Class attribute) The value that is used to initialize the result dataclass and will remain in the results, if
        no result was for a specific attribute in one or more iterations.

    Attributes
    ----------
    inputs_
        List of all input elements that were iterated over.
    raw_results_
        List of all results as dataclass instances.
        The attribute of the dataclass instance will have the value of ``_NOT_SET`` if no result was set.
        To check for this, you can use ``isinstance(val, TypedIterator.NULL_VALUE)``.
    {result_name}_
        The aggregated results for the respective result name.
    done_
        True, if the iterator is done.
        If the iterator is not done, but you try to access the results, a warning will be raised.

    See Also
    --------
    tpcp.misc.TypedIterator
        Generic version of this iterator
    iter_gs
        Functional interface to iterate over gs.

    """

    DEFAULT_AGGREGATIONS: ClassVar[list[tuple[str, _aggregator_type[pd.DataFrame]]]] = [
        ("initial_contacts", create_aggregate_df(["ic"])),
        # TODO: It might be nice for the cadence, sl and gait_speed to actually shift the time values in the index.
        #       However, our cadence time values are in seconds. This makes things tricky, as the aggregator would need
        #       to know the sampling rate of the data.
        ("cadence", create_aggregate_df()),
        ("stride_length", create_aggregate_df()),
        ("gait_speed", create_aggregate_df()),
    ]

    DEFAULT_AGGREGATORS = Aggregators()

    DEFAULT_DATA_TYPE = FullPipelinePerGsResult

    def __init__(
        self,
        data_type: type[DataclassT] = DEFAULT_DATA_TYPE,
        aggregations: Sequence[tuple[str, _aggregator_type[Any]]] = cf(DEFAULT_AGGREGATIONS),
    ) -> None:
        super().__init__(data_type, aggregations)

    def iterate(
        self, data: pd.DataFrame, gs_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[str, pd.DataFrame], DataclassT]]:
        """Iterate over the gait sequences one by one.

        Parameters
        ----------
        data
            The data to iterate over.
        gs_list
            The list of gait-sequences.
            The "start" and "end" columns are expected to match the units of the data index.

        Yields
        ------
        gs_data : tuple[str, pd.DataFrame]
            The data per gait-sequence.
            This is a tuple where the first element is the gait-sequence-id (i.e. the index from the gs-dataframe)
            and the second element is the data cut from the data dataframe.
        result_object
            The empty result object (instance of the provided Dataclass) that should be filled with the results during
            iteration.

        """
        yield from self._iterate(iter_gs(data, gs_list))
