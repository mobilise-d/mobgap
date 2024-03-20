from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Generic, NamedTuple, TypeVar, Union, overload

import pandas as pd
from tpcp import cf
from tpcp.misc import BaseTypedIterator, set_defaults
from typing_extensions import TypeAlias

DataclassT = TypeVar("DataclassT")


class GaitSequence(NamedTuple):
    gs_id: str
    start: int
    end: int

    @property
    def id(self) -> str:
        return self.gs_id


class WalkingBout(NamedTuple):
    wb_id: str
    start: int
    end: int

    @property
    def id(self) -> str:
        return self.wb_id


def iter_gs(
    data: pd.DataFrame, gs_list: pd.DataFrame
) -> Iterator[tuple[Union[GaitSequence, WalkingBout], pd.DataFrame]]:
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
    Union[GaitSequence, WalkingBout]
       A named tuple representing the gait-sequence or walking-bout.
       Note, that only the start, end and id attributes are present.
       If other columns are present in the gs_list, they will be ignored.
       Independent of the "type" a unique id attribute is present, that represents either the ``gs_id`` or
       the ``wb_id``.
    pd.DataFrame
        The data of a single gait-sequence.
        Note, that we don't change the index of the data.
        If the data was using an index that started at the beginning of the recording, the index of the individual
        sequences will still be relative to the beginning of the recording.

    """
    # TODO: Add validation that we passed a valid gs_list
    gs: Union[GaitSequence, WalkingBout]
    gs_list = gs_list.reset_index()
    if "wb_id" in gs_list.columns:
        named_tuple = WalkingBout
        index_col = "wb_id"
    else:
        named_tuple = GaitSequence
        index_col = "gs_id"
    relevant_cols = [index_col, "start", "end"]
    for gs in gs_list[relevant_cols].itertuples(index=False):
        # We explicitly cast GS to the right type to allow for `gs.id` to work.
        yield named_tuple(*gs), data.iloc[gs.start : gs.end]


@dataclass
class FullPipelinePerGsResult:
    """Default expected result type for the gait-sequence iterator.

    When using the :class:`~mobgap.pipeline.GsIterator` with the default configuration, an instance of this dataclass
    will be created for each gait-sequence.

    Each value is expected to be a dataframe.

    Attributes
    ----------
    ic_list
        The initial contacts for each gait-sequence.
        This is a dataframe with a column called ``ic``.
        The values of this ic-column are expected to be samples relative to the start of the gait-sequence.
    cad_per_sec
        The cadence values within each gait-sequence.
        This dataframe has no further requirements relevant for the iterator.
    stride_length
        The stride length values within each gait-sequence.
        This dataframe has no further requirements relevant for the iterator.
    gait_speed
        The gait speed values within each gait-sequence.
        This dataframe has no further requirements relevant for the iterator.

    """

    ic_list: pd.DataFrame
    cad_per_sec: pd.DataFrame
    stride_length: pd.DataFrame
    gait_speed: pd.DataFrame


_inputs_type: TypeAlias = tuple[tuple, pd.DataFrame]

T = TypeVar("T")
_aggregator_type: TypeAlias = Callable[[list[_inputs_type], list[pd.DataFrame]], T]


def create_aggregate_df(
    fix_gs_offset_cols: Sequence[str] = ("start", "end"),
    *,
    fix_gs_offset_index: bool = False,
    _potential_index_names: Sequence[str] = ("wb_id", "gs_id"),
) -> _aggregator_type[pd.DataFrame]:
    """Create an aggregator for the GS iterator that aggregates dataframe results into a single dataframe.

    The aggregator will also fix the offset of the given columns by adding the start value of the gait-sequence.
    This way the final dataframe will have all sample based time-values relative to the start of the recording.

    Parameters
    ----------
    fix_gs_offset_cols
        The columns that should be adapted to be relative to the start of the recording.
        By default, this is ``("start", "end")``.
        If you don't want to fix any columns, you can set this to an empty list.
    fix_gs_offset_index
        If True, the index of the dataframes will be adapted to be relative to the start of the recording.
        This only makes sense, if the index represents sample values relative to the start of the gs.
    _potential_index_names
        The potential names of the index columns.
        This usually does not need to be changed.

    """
    if len(_potential_index_names) == 0:
        raise ValueError("You need to provide at least one potential index name.")

    def aggregate_df(inputs: list[_inputs_type], outputs: list[pd.DataFrame]) -> pd.DataFrame:
        sequences, _ = zip(*inputs)

        for iter_index_name in _potential_index_names:
            if iter_index_name in sequences[0]._fields:
                break

        to_concat = {}
        for gs, o in zip(sequences, outputs):
            if not isinstance(o, pd.DataFrame):
                raise TypeError(f"Expected dataframe for this aggregator, but got {type(o)}")
            o = o.copy()  # noqa: PLW2901
            if fix_gs_offset_cols:
                cols_to_fix = set(fix_gs_offset_cols).intersection(o.columns)
                o[list(cols_to_fix)] += gs.start
            if fix_gs_offset_index:
                o.index += gs.start
            to_concat[gs[0]] = o

        return pd.concat(to_concat, names=[iter_index_name, *outputs[0].index.names])

    return aggregate_df


class GsIterator(BaseTypedIterator[DataclassT], Generic[DataclassT]):
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
        By default, this is ``GsIterator.DEFAULT_DATA_TYPE``, which should handle all typical results of a gait analysis
        pipeline.
    aggregations
        An optional list of aggregations to apply to the results.
        This has the form ``[(result_name, aggregation_function), ...]``.
        If a result-name is in the list, the aggregation will be applied to it, when accessing the respective result
        attribute (i.e. ``{result_name}_``).
        If no aggregation is defined for a result, a simple list of all results will be returned.
        By default, this is ``GsIterator.DEFAULT_AGGREGATIONS``.
    NULL_VALUE
        (Class attribute) The value that is used to initialize the result dataclass and will remain in the results, if
        no result was for a specific attribute in one or more iterations.
    PredefinedParameters
        (Class attribute) Predefined parameters that can be used depending on which aggregation you want to use.
        In all provided cases the ``data_type`` is set to :class:`FullPipelinePerGsResult`.
        This datatype provides the following attributes:

        - ``ic_list`` (pd.DataFrame with a column called ``ic``): The initial contacts for each gait-sequence.
        - ``cad_per_sec`` (pd.DataFrame): The cadence values within each gait-sequence.
        - ``stride_length`` (pd.DataFrame): The stride length values within each gait-sequence.
        - ``gait_speed`` (pd.DataFrame): The gait speed values within each gait-sequence.
    DefaultAggregators
        (Class attribute) Class that holds some aggregator functions that can be used to create custom aggregations.

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

    class PredefinedParameters:
        """Predefined parameters for the gait-sequence iterator.

        Attributes
        ----------
        default_aggregation
            The default of the TypedIterator using the :class:`FullPipelinePerGsResult` as data_type and trying to
            aggregate all results so that the time values in the final outputs are relative to the start of the
            recording.
        default_aggregation_rel_to_gs
            Same as ``default_aggregation``, but the time values in the final outputs are relative to the start of the
            respective gait-sequence (i.e. no modification of the time values is done).

        """

        default_aggregation: ClassVar[dict[str, Any]] = {
            "data_type": FullPipelinePerGsResult,
            "aggregations": cf(
                [
                    ("ic_list", create_aggregate_df(["ic"])),
                    ("cad_per_sec", create_aggregate_df([], fix_gs_offset_index=True)),
                    ("stride_length", create_aggregate_df()),
                    ("gait_speed", create_aggregate_df()),
                ]
            ),
        }
        default_aggregation_rel_to_gs: ClassVar[dict[str, Any]] = {
            "data_type": FullPipelinePerGsResult,
            "aggregations": cf(
                [
                    ("ic_list", create_aggregate_df([])),
                    ("cad_per_sec", create_aggregate_df([])),
                    ("stride_length", create_aggregate_df([])),
                    ("gait_speed", create_aggregate_df([])),
                ]
            ),
        }

    class DefaultAggregators:
        """Available aggregators for the gait-sequence iterator.

        Note, that all of them are constructors for aggregators, as they have some configuration options.
        To use them as aggregators, you need to call them with the desired configuration.

        Examples
        --------
        >>> from mobgap.pipeline import GsIterator
        >>> my_aggregation = [("my_result", GsIterator.DefaultAggregators.create_aggregate_df(["my_col"]))]
        >>> iterator = GsIterator(aggregations=my_aggregation)

        """

        create_aggregate_df = create_aggregate_df

    # We provide this explicit overload, so that the type of the default value is correcttly inferred.
    # This way there is not need to "bind" FullPipelinePerGsResult on init, when the defaults are used.
    @overload
    def __init__(
        self: "GsIterator[FullPipelinePerGsResult]",
        data_type: type[FullPipelinePerGsResult] = ...,
        aggregations: Sequence[tuple[str, _aggregator_type[Any]]] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: "GsIterator[DataclassT]",
        data_type: type[DataclassT] = ...,
        aggregations: Sequence[tuple[str, _aggregator_type[Any]]] = ...,
    ) -> None: ...

    @set_defaults(**PredefinedParameters.default_aggregation)
    def __init__(
        self,
        data_type,
        aggregations,
    ) -> None:
        super().__init__(data_type, aggregations)

    def iterate(
        self, data: pd.DataFrame, gs_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[Union[WalkingBout, GaitSequence], pd.DataFrame], DataclassT]]:
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
