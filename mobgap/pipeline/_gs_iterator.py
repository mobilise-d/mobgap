from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
    overload,
)

import pandas as pd
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import TypeAlias

from mobgap.pipeline._overwrite_typed_iterator import BaseTypedIterator, TypedIteratorResultTuple, _NotSet

DataclassT = TypeVar("DataclassT")


class GaitSequence(NamedTuple):
    """A simple tuple representing a Gait Sequence."""

    id: str
    start: int
    end: int


def iter_gs(data: pd.DataFrame, gs_list: pd.DataFrame) -> Iterator[tuple[GaitSequence, pd.DataFrame]]:
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
    GaitSequence
       A named tuple representing the gait-sequence or walking-bout.
       Note, that only the start, end and id attributes are present.
       If other columns are present in the gs_list, they will be ignored.
       Independent of the "type" a unique ``id`` attribute is present, that represents either the ``gs_id`` or
       the ``wb_id``.
    pd.DataFrame
        The data of a single gait-sequence.
        Note, that we don't change the index of the data.
        If the data was using an index that started at the beginning of the recording, the index (aka ``.loc``) of the
        individual sequences will still be relative to the beginning of the recording.
        The first sample in the returned data (aka ``.iloc``) will correcly correspond to the first sample of the GS.

    """
    # TODO: Add validation that we passed a valid gs_list
    gs: GaitSequence
    gs_list = gs_list.reset_index()
    if "wb_id" in gs_list.columns:
        index_col = "wb_id"
    else:
        index_col = "gs_id"
    relevant_cols = [index_col, "start", "end"]
    for gs in gs_list[relevant_cols].itertuples(index=False):
        # We explicitly cast GS to the right type to allow for `gs.id` to work.
        yield GaitSequence(*gs), data.iloc[gs.start : gs.end]


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


InputType: TypeAlias = tuple[GaitSequence, pd.DataFrame]
ResultT = TypeVar("ResultT")
# TODO: Move and adjust that type alias
GsIteratorResult: TypeAlias = TypedIteratorResultTuple[InputType, FullPipelinePerGsResult]
GsIteratorResultT: TypeAlias = TypedIteratorResultTuple[InputType, ResultT]


T = TypeVar("T")
_aggregator_type: TypeAlias = Callable[[list[GsIteratorResult]], T]


def create_aggregate_df(
    result_name: str,
    fix_gs_offset_cols: Sequence[str] = ("start", "end"),
    *,
    fix_gs_offset_index: bool = False,
    _null_value: _NotSet = BaseTypedIterator.NULL_VALUE,
) -> _aggregator_type[pd.DataFrame]:
    """Create an aggregator for the GS iterator that aggregates dataframe results into a single dataframe.

    The aggregator will also fix the offset of the given columns by adding the start value of the gait-sequence.
    This way the final dataframe will have all sample based time-values relative to the start of the recording.

    Parameters
    ----------
    result_name
        The name of the result key within the result object, the aggregation is applied to
    fix_gs_offset_cols
        The columns that should be adapted to be relative to the start of the recording.
        By default, this is ``("start", "end")``.
        If you don't want to fix any columns, you can set this to an empty list.
    fix_gs_offset_index
        If True, the index of the dataframes will be adapted to be relative to the start of the recording.
        This only makes sense, if the index represents sample values relative to the start of the gs.
    _null_value
        A fixed value that should indicate that no results were produced.
        You don't need to change this, unless you are doing very advanced stuff.

    Notes
    -----
    Fixing the offset works by getting the start value of the gait-sequence and adding it to the respective columns.
    This is "easy" for the main iteration, where the gait-sequences contains all the relevant information.
    For sub-iteration, we need to consider the parent context.
    For this, the GS-Iterator, places the parent gait-sequence in the iteration context.

    """

    def aggregate_df(values: list[GsIteratorResult]) -> pd.DataFrame:
        non_null_results: list[GsIteratorResultT[pd.DataFrame]] = GsIterator.filter_iterator_results(
            values, result_name, _null_value
        )
        if len(non_null_results) == 0:
            # Note: We don't have a way to properly know the names of the index cols or the cols themselve here...
            return pd.DataFrame()

        # We assume that all elements have the same iteration context.
        iter_index_name = non_null_results[0].iteration_context.get("id_col_name", "gs_id")
        if not isinstance(iter_index_name, list):
            iter_index_name = [iter_index_name]

        to_concat = {}
        for rt in non_null_results:
            df = rt.result
            gs_id = rt.input[0].id
            offset = rt.input[0].start

            parent_gs: Optional[GaitSequence] = rt.iteration_context.get("parent_gs", None)

            if rt.iteration_name == "__sub_iter__":
                if not parent_gs:
                    raise RuntimeError("Sub-iteration without parent GS.")
                offset += parent_gs.start
                gs_id = (parent_gs.id, gs_id)
            elif rt.iteration_name == "__main__":
                if parent_gs:
                    raise RuntimeError("Main iteration with parent GS should not exist.")
            else:
                raise RuntimeError("Unexpected iteration type")

            df = df.copy()
            if fix_gs_offset_cols:
                cols_to_fix = set(fix_gs_offset_cols).intersection(df.columns)
                df[list(cols_to_fix)] += offset
            if fix_gs_offset_index:
                df.index += offset
            to_concat[gs_id] = df

        return pd.concat(to_concat, names=[*iter_index_name, *next(iter(to_concat.values())).index.names])

    return aggregate_df


class GsIterator(BaseTypedIterator[InputType, DataclassT], Generic[DataclassT]):
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
        no result was provided for a specific attribute in one or more iterations.
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
    IteratorResult
        (Class attribute) Type alias for the resultype of the iterator. ``raw_results_`` will be a list of these.
        Note, that when using this outside of the class, this type will be a generic without a type for the ``result``
        field.
        You need to bind it as ``GsIterator.IteratorResult[MyCustomResultType]`` to get the correct type.
        This will then be the correct result type of an iterator using the same ``data_type`` (i.e.
        ``gs_iterator = GsIterator[MyCustomResultType](MyCustomResultType)``).

    Attributes
    ----------
    results_
        The actual aggregated results.
        This is an instance of the provided custom result type (``data_type``).
        This makes it easy to access the individual result attributes.
        Each value will be the result of the aggregation function registered for the specific field.
        If no aggregation functions exist, simply a list of the result values for this field is provided.
        The results can only be accessed once the iteration is done.
    raw_results_
        # TODO: Adapt path once tpcp PR is merged
        List of all results as ``TypedIteratorResultTuple`` instances.
        This is the input to the aggregation functions.
        The attribute of the ``result`` dataclass instance will have the value of ``_NOT_SET`` if no result was set.
        To check for this, you can use ``isinstance(val, TypedIterator.NULL_VALUE)`` or the
        ``filter_iterator_results`` method to remove all results with a ``NULL_VALUE``.
    done_
        A dictionary indicating of a specific iterator is done.
        This can have the keys ``__main__`` or ``__sub_iter`` for the main iteration triggered by ``iterate`` or
        sub-iterations triggered by ``iterate_subregions`` or ``with_subregion``.
        The value will be ``True`` if the respective iteration is done, ``False`` if it is currently running and
        missing if it was never started.
        If the iterator is not done, but you try to access the results, an error will be raised.

    See Also
    --------
    tpcp.misb.BaseTypedIterator
        Baseclass of this iterator
    tpcp.misc.TypedIterator
        Generic version of this iterator
    iter_gs
        Functional interface to iterate over gs.

    """

    # This is required to correctly interfere the new bound type
    IteratorResult: TypeAlias = TypedIteratorResultTuple[InputType, DataclassT]

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
                    ("ic_list", create_aggregate_df("ic_list", ["ic"])),
                    ("cad_per_sec", create_aggregate_df("cad_per_sec", [], fix_gs_offset_index=True)),
                    ("stride_length", create_aggregate_df("stride_length")),
                    ("gait_speed", create_aggregate_df("gait_speed")),
                ]
            ),
        }
        default_aggregation_rel_to_gs: ClassVar[dict[str, Any]] = {
            "data_type": FullPipelinePerGsResult,
            "aggregations": cf(
                [
                    ("ic_list", create_aggregate_df("ic_list", [])),
                    ("cad_per_sec", create_aggregate_df("cad_per_sec", [])),
                    ("stride_length", create_aggregate_df("stride_length", [])),
                    ("gait_speed", create_aggregate_df("gait_speed", [])),
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
        >>> my_aggregation = [
        ...     (
        ...         "my_result",
        ...         GsIterator.DefaultAggregators.create_aggregate_df("my_result", fix_gs_offset_cols=["my_col"]),
        ...     )
        ... ]
        >>> iterator = GsIterator(aggregations=my_aggregation)

        """

        create_aggregate_df = create_aggregate_df

    # We provide this explicit overload, so that the type of the default value is correcttly inferred.
    # This way there is not need to "bind" FullPipelinePerGsResult on init, when the defaults are used.
    @overload
    def __init__(
        self: "GsIterator[FullPipelinePerGsResult]",
        data_type: type[FullPipelinePerGsResult] = ...,
        aggregations: Sequence[tuple[str, Callable[[list[IteratorResult]], Any]]] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: "GsIterator[DataclassT]",
        data_type: type[DataclassT] = ...,
        aggregations: Sequence[tuple[str, Callable[[list[IteratorResult]], Any]]] = ...,
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
    ) -> Iterator[tuple[tuple[GaitSequence, pd.DataFrame], DataclassT]]:
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
        context = {"id_col_name": "wb_id" if "wb_id" in gs_list.index.names else "gs_id"}
        yield from self._iterate(iter_gs(data, gs_list), iteration_context=context)

    def iterate_subregions(
        self, gs_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[GaitSequence, pd.DataFrame], DataclassT]]:
        # We only allow sub iterations, when there are no other subiterations running.
        if self.done_.get("__main__", True):
            raise ValueError("Sub-iterations can only be started, when the main iteration is still running")
        if not self.done_.get("__sub_iter__", True):
            raise ValueError("Sub-iterations are not allowed within sub-iterations.")

        current_result = self._raw_results[-1]
        current_gs, current_data = current_result.input
        id_col_names = [current_result.iteration_context["id_col_name"], "sub_gs_id"]
        yield from self._iterate(
            iter_gs(current_data, gs_list),
            iteration_name="__sub_iter__",
            iteration_context={"id_col_name": id_col_names, "parent_gs": current_gs},
        )

    def with_subregion(self, gs_list: pd.DataFrame) -> tuple[tuple[GaitSequence, pd.DataFrame], DataclassT]:
        if len(gs_list) != 1:
            raise ValueError(
                "``with_subregions`` can only be used with single-subregions. "
                "However, the passed ``gs_list`` has 0 or more than one GSs. "
                "If you want to process multiple sub-regions, use ``iterate_subregions``."
            )
        return list(self.iterate_subregions(gs_list))[0]
