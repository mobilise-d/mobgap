from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
    overload,
)

import pandas as pd
from tpcp import cf
from tpcp.misc import BaseTypedIterator, TypedIteratorResultTuple, custom_hash, set_defaults
from tpcp.misc._typed_iterator import _NotSet
from typing_extensions import TypeAlias


class Region(NamedTuple):
    """A simple tuple representing a Gait Sequence."""

    id: str
    start: int
    end: int
    id_origin: Optional[str] = None


class RegionDataTuple(NamedTuple):
    """A simple tuple representing the input of the Gait Sequence Iterator."""

    region: Region
    data: pd.DataFrame


T = TypeVar("T")
DataclassT = TypeVar("DataclassT")


def _infer_id_col(region_list: pd.DataFrame, id_col: Optional[str] = None) -> str:
    """Infer the id column from the given gait-sequence list.

    Parameters
    ----------
    region_list
        The gait-sequence/region_list list.
    id_col
        The name of the column/index level that should be used as the id of the returned Region objects.
        If None, we try to automatically infer the name.
        If the input contains ``wb_id`` or ``gs_id``, we will use this.
        When a single index column exists, we will use this.
        If multiple index columns exist, we will raise an error and you need to specify the column.

    Returns
    -------
    str
        The name of the column/index level that should be used as the id of the returned Region objects.

    """
    if id_col:
        return id_col
    region_list_all_cols = region_list.reset_index().columns
    if "wb_id" in region_list_all_cols:
        return "wb_id"
    if "gs_id" in region_list_all_cols:
        return "gs_id"
    if len(region_list.index.names) == 1 and (name := region_list.index.names[0]) is not None:
        return name
    raise ValueError(
        "Could not infer the id column from the gait-sequence list. "
        "Please specify the column/index level that should be used as the id."
    )


def iter_gs(
    data: pd.DataFrame, region_list: pd.DataFrame, *, id_col: Optional[str] = None
) -> Iterator[tuple[Region, pd.DataFrame]]:
    """Iterate over the data based on the given gait-sequences.

    This will yield a dataframe for each gait-sequence.
    We assume that the gait-sequences are sorted by start time and contain a ``start`` and ``end`` column that match
    the units of the data index.

    Parameters
    ----------
    data
        The data to iterate over.
    region_list
        The list of gait-sequences.
    id_col
        The name of the column/index level that should be used as the id of the returned Region objects.
        If None, we try to automatically infer the name.
        If the input contains ``wb_id`` or ``gs_id``, we will use this.
        When a single index column exists, we will use this.
        If multiple index columns exist, we will raise an error and you need to specify the column.

    Yields
    ------
    Region
       A named tuple representing the gait-sequence or walking-bout.
       Note, that only the start, end and id attributes are present.
       If other columns are present in the region_list, they will be ignored.
       Independent of the "type" a unique ``id`` attribute is present, that represents either the ``gs_id`` or
       the ``wb_id``.
    pd.DataFrame
        The data of a single gait-sequence/region.
        Note, that we don't change the index of the data.
        If the data was using an index that started at the beginning of the recording, the index (aka ``.loc``) of the
        individual sequences will still be relative to the beginning of the recording.
        The first sample in the returned data (aka ``.iloc``) will correctly correspond to the first sample of the GS.

    """
    # TODO: Add validation that we passed a valid region_list
    gs: Region
    index_col = _infer_id_col(region_list, id_col)
    region_list = region_list.reset_index()
    relevant_cols = [index_col, "start", "end"]

    # Perform checks on the entire DataFrame
    if (region_list["start"] < 0).any():
        raise ValueError("The start of a gait-sequence should not be negative.")
    if (region_list["end"] < region_list["start"]).any():
        raise ValueError("The end of a gait-sequence should be larger than the start.")
    if (region_list["end"] > len(data)).any():
        raise ValueError("The end of a gait-sequence should not be larger than the length of the data.")

    for gs in region_list[relevant_cols].itertuples(index=False):
        # We explicitly cast GS to the right type to allow for `gs.id` to work.
        yield RegionDataTuple(Region(*gs, index_col), data.iloc[gs.start : gs.end])


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
    turn_list
        The turn list for each gait-sequence.
        The dataframe has at least columns called ``start`` and ``end``.
        The values of these columns are expected to be samples relative to the start of the gait-sequence.
    cadence_per_sec
        The cadence values within each gait-sequence.
        This dataframe has no further requirements relevant for the iterator.
    stride_length_per_sec
        The stride length values within each gait-sequence.
        This dataframe has no further requirements relevant for the iterator.
    walking_speed_per_sec
        The gait speed values within each gait-sequence.
        This dataframe has no further requirements relevant for the iterator.

    """

    ic_list: pd.DataFrame
    turn_list: pd.DataFrame
    cadence_per_sec: pd.DataFrame
    stride_length_per_sec: pd.DataFrame
    walking_speed_per_sec: pd.DataFrame


def _build_id_cols(region: Region, parent_region: Optional[Region]) -> list[str]:
    iter_index_name = [region.id_origin]
    if parent_region is not None:
        iter_index_name = [parent_region.id_origin, *iter_index_name]
    return iter_index_name


def _validate_iter_type(iter_type: str, parent_region: Optional[Region]) -> None:
    if iter_type not in ["__sub_iter__", "__main__"]:
        raise RuntimeError("Unexpected iteration type")
    if parent_region and iter_type == "__main__":
        raise RuntimeError("Main iteration with parent region should not exist.")
    if not parent_region and iter_type == "__sub_iter__":
        raise RuntimeError("Sub-iteration without parent region.")


def create_aggregate_df(
    result_name: str,
    fix_offset_cols: Sequence[str] = ("start", "end"),
    *,
    fix_offset_index: bool = False,
    _null_value: _NotSet = BaseTypedIterator.NULL_VALUE,
) -> Callable[[list["GsIterator.IteratorResult[Any]"]], T][pd.DataFrame]:
    """Create an aggregator for the GS iterator that aggregates dataframe results into a single dataframe.

    The aggregator will also fix the offset of the given columns by adding the start value of the gait-sequence.
    This way the final dataframe will have all sample based time-values relative to the start of the recording.

    Parameters
    ----------
    result_name
        The name of the result key within the result object, the aggregation is applied to
    fix_offset_cols
        The columns that should be adapted to be relative to the start of the recording.
        By default, this is ``("start", "end")``.
        If you don't want to fix any columns, you can set this to an empty list.
    fix_offset_index
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

    def aggregate_df(values: list["GsIterator.IteratorResult[Any]"]) -> pd.DataFrame:
        non_null_results: list[GsIterator.IteratorResult[pd.DataFrame]] = GsIterator.filter_iterator_results(
            values, result_name, _null_value
        )
        if len(non_null_results) == 0:
            # Note: We don't have a way to properly know the names of the index cols or the cols themselve here...
            return pd.DataFrame()

        # We assume that all elements have the same iteration context.
        first_element = non_null_results[0]
        iter_index_name = _build_id_cols(
            first_element.input.region, first_element.iteration_context.get("parent_region", None)
        )

        to_concat = {}
        for rt in non_null_results:
            df = rt.result
            region_id, offset, *_ = rt.input.region

            parent_region: Optional[Region] = rt.iteration_context.get("parent_region", None)

            _validate_iter_type(rt.iteration_name, parent_region)

            if parent_region:
                offset += parent_region.start
                region_id = (parent_region.id, region_id)

            df = df.copy()
            if fix_offset_cols:
                cols_to_fix = set(fix_offset_cols).intersection(df.columns)
                df[list(cols_to_fix)] += offset
            if fix_offset_index:
                df.index += offset
            to_concat[region_id] = df

        return pd.concat(to_concat, names=[*iter_index_name, *next(iter(to_concat.values())).index.names])

    return aggregate_df


class GsIterator(BaseTypedIterator[RegionDataTuple, DataclassT], Generic[DataclassT]):
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
        Each aggregation function gets ``raw_results_`` provided as input and can return an arbitrary object.
        If a result-name is in the list, the aggregation will be applied to it, when accessing the ``results_``
        (i.e. ``results_.{result_name}``).
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
        (Class attribute) Type alias for the result-type of the iterator. ``raw_results_`` will be a list of these.
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
        To check for this, you can use ``isinstance(val, GsIterator.NULL_VALUE)`` or the
        ``GsIterator.filter_iterator_results`` method to remove all results with a ``NULL_VALUE``.
    done_
        A dictionary indicating of a specific iterator is done.
        This can have the keys ``__main__`` or ``__sub_iter`` for the main iteration triggered by ``iterate`` or
        sub-iterations triggered by ``iterate_subregions`` or ``with_subregion``.
        The value will be ``True`` if the respective iteration is done, ``False`` if it is currently running and
        missing if it was never started.
        If the main iterator is not done, but you try to access the results, an error will be raised.

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
    IteratorResult: TypeAlias = TypedIteratorResultTuple[RegionDataTuple, DataclassT]

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

        default_aggregation: Final = MappingProxyType(
            {
                "data_type": FullPipelinePerGsResult,
                "aggregations": cf(
                    [
                        ("ic_list", create_aggregate_df("ic_list", ["ic"])),
                        ("turn_list", create_aggregate_df("turn_list", ["start", "end", "center"])),
                        ("cadence_per_sec", create_aggregate_df("cadence_per_sec", [], fix_offset_index=True)),
                        (
                            "stride_length_per_sec",
                            create_aggregate_df("stride_length_per_sec", [], fix_offset_index=True),
                        ),
                        (
                            "walking_speed_per_sec",
                            create_aggregate_df("walking_speed_per_sec", [], fix_offset_index=True),
                        ),
                    ]
                ),
            }
        )
        default_aggregation_rel_to_gs: Final = MappingProxyType(
            {
                "data_type": FullPipelinePerGsResult,
                "aggregations": cf(
                    [
                        ("ic_list", create_aggregate_df("ic_list", [])),
                        ("turn_list", create_aggregate_df("turn_list", [])),
                        ("cadence_per_sec", create_aggregate_df("cadence_per_sec", [])),
                        ("stride_length_per_sec", create_aggregate_df("stride_length", [])),
                        ("walking_speed_per_sec", create_aggregate_df("gait_speed", [])),
                    ]
                ),
            }
        )

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
        ...         GsIterator.DefaultAggregators.create_aggregate_df(
        ...             "my_result", fix_offset_cols=["my_col"]
        ...         ),
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
        self, data: pd.DataFrame, region_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[Region, pd.DataFrame], DataclassT]]:
        """Iterate over the gait sequences one by one.

        Parameters
        ----------
        data
            The data to iterate over.
        region_list
            The list of gait-sequences.
            The "start" and "end" columns are expected to match the units of the data index.

        Yields
        ------
        region_data : tuple[Region, pd.DataFrame]
            The data per gait-sequence.
            This is a tuple where the first element is a ``Region`` object that contains the relevant information
            about the current GS/WB/region and the second element is the data cut from the data dataframe.
        result_object
            The empty result object (instance of the provided Dataclass) that should be filled with the results during
            iteration.

        """
        yield from self._iterate(iter_gs(data, region_list))

    def iterate_subregions(
        self, sub_region_list: pd.DataFrame
    ) -> Iterator[tuple[tuple[Region, pd.DataFrame], DataclassT]]:
        """Iterate subregions within the current gait sequence.

        This can be called within the for-loop created by the main iteration to trigger the iteration over subregions.
        The provided subregions are expected to be relative to the current gait-sequence.
        Working with subregions, can be a little tricky, and we recommend you read through the respective pipeline
        examples to avoid foot-guns.

        .. note:: If you only have a single GS in your ``sub_region_list`` you can also use the ``with_subregion``
                  method and avoid creating a nested for-loop.

        Parameters
        ----------
        sub_region_list
            The list of subregions within the current region.
            The "start" and "end" values need to be relative to the current gait sequence the parent is iterating over.

        Returns
        -------
        region_data : tuple[Region, pd.DataFrame]
            The data per gait-sequence.
            This is a tuple where the first element is a ``Region`` object that contains the relevant information
            about the current GS/WB/region and the second element is the data cut from the data dataframe.
        result_object
            The empty result object (instance of the provided Dataclass) that should be filled with the results during
            iteration.

        """
        # We only allow sub iterations, when there are no other subiterations running.
        if getattr(self, "done_", {}).get("__main__", True):
            raise ValueError("Sub-iterations can only be started, when the main iteration is still running")
        if not self.done_.get("__sub_iter__", True):
            raise ValueError("Sub-iterations are not allowed within sub-iterations.")

        current_result = self._raw_results[-1]
        current_region, current_data = current_result.input

        # We calculate the hash of the last outer result to check if it was changed during the sub-iteration.
        # Note, that when you are using the ``subregion`` context manager, this check is duplicated.
        # The reason for that is that with the context manager, we have a clear entry and exist point that we would
        # not otherwise have, when we simply iterate a single subregion.
        current_result_obj = current_result.result
        before_result_hash = custom_hash(current_result_obj)

        yield from self._iterate(
            iter_gs(current_data, sub_region_list),
            iteration_name="__sub_iter__",
            iteration_context={"parent_region": current_region},
        )

        after_result_hash = custom_hash(current_result_obj)
        if before_result_hash != after_result_hash:
            raise RuntimeError(
                "It looks like you accessed the result of the main iteration within the subregion iteration. "
                "This might lead to unexpected results. "
                "Make sure you use the result object returned by the subregion iteration."
            )

    def with_subregion(self, sub_region_list: pd.DataFrame) -> tuple[tuple[Region, pd.DataFrame], DataclassT]:
        """Get a subregion of the current gait sequence.

        For details see ``iterate_subregions``.

        Parameters
        ----------
        sub_region_list
            A region list containing a SINGLE subregion (i.e. on row) within the current region.
            The "start" and "end" values need to be relative to the current gait sequence the parent is iterating over.
            For the ``with_subregions`` method this must be just a single GS.
            If you want to iterate multiple GSs see ``iterate_subregions``.

        Returns
        -------
        inputs
            A tuple with a gait-sequence object and the data corresponding to the subregion.
        result_object
            An empty result object for the subregion that can be used to provide results for it.

        Notes
        -----
        Internally, this simply uses ``iterate_subregions``, but completes the iteration over the single GS and returns
        it.

        Examples
        --------
        >>> gs_list = pd.DataFrame({"start": [0, 10, 20], "end": [10, 20, 30]}).rename_axis(
        ...     "gs_id"
        ... )
        >>> gs_iterator = GsIterator()
        >>> for (gs, data), r in gs_iterator.iterate(data, gs_list):
        ...     sub_region = pd.DataFrame(
        ...         {"start": [3], "end": [len(data) - 3]}
        ...     ).rename_axis("gs_id")
        ...     (sub_gs, sub_data), sub_r = gs_iterator.with_subregion(sub_region)
        ...     # Do something with the subregion data
        ...     sub_r.my_result = pd.DataFrame({"my_col": [1, 2, 3]})

        """
        if len(sub_region_list) != 1:
            raise ValueError(
                "``with_subregions`` can only be used with single-subregions. "
                "However, the passed ``region_list`` has 0 or more than one GSs. "
                "If you want to process multiple sub-regions, use ``iterate_subregions``."
            )
        return list(self.iterate_subregions(sub_region_list))[0]  # noqa: RUF015

    # Note: Iterator[...] is the correct type annotation here. PyCharm just does not recognize it. See:
    #       PY-71674 PyCharm doesn't infer types when using contextlib.contextmanager decorator on a method
    @contextmanager
    def subregion(self, sub_region_list: pd.DataFrame) -> Iterator[tuple[tuple[Region, pd.DataFrame], DataclassT]]:
        """Context manager for handling a subregion of the current gait sequence.

        This is basically just syntactic sugar for the ``with_subregion`` method.
        However, it also performs a check that you are only writing to the intended result object while within the
        ``with`` block, which hopefully prevents some mistakes.

        Parameters
        ----------
        sub_region_list
            The list of subregions within the current gait-sequence.
            The "start" and "end" values need to be relative to the current gait sequence the parent is iterating over.
            For the ``with_subregions`` method this must be just a single GS.
            If you want to iterate multiple GSs see ``iterate_subregions``.

        Yields
        ------
        inputs
            A tuple with a gait-sequence object and the data corresponding to the subregion.
        result_object
            An empty result object for the subregion that can be used to provide results for it.

        Notes
        -----
        Internally, this simply uses ``iterate_subregions``, but completes the iteration over the single GS and returns
        it.

        Examples
        --------
        >>> gs_list = pd.DataFrame({"start": [0, 10, 20], "end": [10, 20, 30]}).rename_axis(
        ...     "gs_id"
        ... )
        >>> gs_iterator = GsIterator()
        >>> for (gs, data), r in gs_iterator.iterate(data, gs_list):
        ...     sub_region = pd.DataFrame(
        ...         {"start": [3], "end": [len(data) - 3]}
        ...     ).rename_axis("gs_id")
        ...     with gs_iterator.subregion(sub_region) as ((sub_gs, sub_data), sub_r):
        ...         # Do something with the subregion data
        ...         sub_r.my_result = pd.DataFrame({"my_col": [1, 2, 3]})

        """
        # TODO: We could use that in the future to perform some more checks. For example we could block writing to the
        #  subregion result after the context is done.
        outer_result = self._raw_results[-1].result
        before_result_hash = custom_hash(outer_result)

        try:
            yield self.with_subregion(sub_region_list)
        finally:
            after_result_hash = custom_hash(outer_result)
            if before_result_hash != after_result_hash:
                raise RuntimeError(
                    "It looks like you accessed the old result object of the main iteration within the subregion "
                    "context. "
                    "Use the result object returned by the context manager!"
                )
