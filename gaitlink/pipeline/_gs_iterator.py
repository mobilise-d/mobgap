from collections.abc import Iterator
from typing import Generic, TypeVar

import pandas as pd
from tpcp.misc import BaseTypedIterator

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

    """
    for gs in gs_list.itertuples(index=True, name="gs"):
        yield gs[0], data.iloc[gs.start : gs.end]


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
