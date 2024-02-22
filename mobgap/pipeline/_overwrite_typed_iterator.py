import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Generic, TypeVar

from tpcp import Algorithm, cf

DataclassT = TypeVar("DataclassT")
T = TypeVar("T")


class _NotSet:
    def __repr__(self):
        return "_NOT_SET"


class BaseTypedIterator(Algorithm, Generic[DataclassT]):
    """A Base class to implement custom typed iterators.

    This class is missing the ``iterate`` method, which needs to be implemented by the child class.
    It has a `_iterate` method though that does most of the heavy lifting.
    The actual iterate method should handle turning the inputs into an iterable and then call `_iterate`.

    :class:`TypedIterator` provides a "dummy" implementation that expects any type of iterator for the iterate method.
    Custom base classes could provide more elaborate preprocessing of the inputs before iteration.
    For example, cutting sections out of a dataframe based on a list of start and end indices, and then iterating over
    the cut sections.

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
        The attribute of the dataclass instance will have the a value of ``_NOT_SET`` if no result was set.
        To check for this, you can use ``isinstance(val, TypedIterator.NULL_VALUE)``.
    results_
        An instance of the result object with either the "inverted" results (i.e. a list of all results) per attribute
        or the aggregated results (depending on the aggregation function).
        Note, that the typing of the attributes of the result object will not be correct.
    {result_name}_
        The aggregated results for the respective result name.
    done_
        True, if the iterator is done.
        If the iterator is not done, but you try to access the results, a warning will be raised.

    """

    data_type: type[DataclassT]
    aggregations: Sequence[tuple[str, Callable[[list, list], Any]]]

    _raw_results: list[DataclassT]
    _result_fields: set[str]
    done_: bool
    inputs_: list

    NULL_VALUE = _NotSet()

    def __init__(
        self, data_type: type[DataclassT], aggregations: Sequence[tuple[str, Callable[[list, list], Any]]] = cf([])
    ):
        self.data_type = data_type
        self.aggregations = aggregations

    def _iterate(self, iterable: Iterable[T]) -> Iterator[tuple[T, DataclassT]]:
        """Iterate over the given iterable and yield the input and a new empty result object for each iteration.

        Parameters
        ----------
        iterable
            The iterable to iterate over.

        Yields
        ------
        input, result_object
            The input and a new empty result object.
            The result object is a dataclass instance of the type defined in ``self.data_type``.
            All values of the result object are set to ``TypedIterator.NULL_VALUE`` by default.

        """
        if not is_dataclass(self.data_type):
            raise TypeError(f"Expected a dataclass as data_type, got {self.data_type}")

        result_field_names = {f.name for f in fields(self.data_type)}
        not_allowed_fields = {"results", "raw_results", "done", "inputs"}
        if not_allowed_fields.intersection(result_field_names):
            raise ValueError(
                f"The result dataclass cannot have a field called {not_allowed_fields}. "
                "These fields are used by the TypedIterator to store the results. "
                "Having these fields in the result object will result in naming conflicts."
            )

        self._result_fields = result_field_names

        self._raw_results = []
        self.inputs_ = []
        self.done_ = False
        for d in iterable:
            result_object = self._get_new_empty_object()
            self._report_new_result(result_object)
            self._report_new_input(d)
            yield d, result_object
        self.done_ = True
        self.results_ = self.data_type(**{k.name: self._agg_result(k.name) for k in fields(self.data_type)})

    def _report_new_result(self, result: DataclassT):
        self._raw_results.append(result)

    def _report_new_input(self, input: T):
        self.inputs_.append(input)

    def _get_new_empty_object(self) -> DataclassT:
        init_dict = {k.name: self.NULL_VALUE for k in fields(self.data_type)}
        return self.data_type(**init_dict)

    @property
    def raw_results_(self) -> list[DataclassT]:
        if not self.done_:
            warnings.warn("The iterator is not done yet. The results might not be complete.", stacklevel=1)

        return self._raw_results

    def _agg_result(self, name):
        values = [getattr(r, name) for r in self.raw_results_]
        # if an aggregator is defined for the specific item, we apply it
        aggregations = dict(self.aggregations)
        if name in aggregations and all(v is not self.NULL_VALUE for v in values):
            return aggregations[name](self.inputs_, values)
        return values

    def __getattr__(self, item):
        # We assume a correct result name ends with an underscore
        if (actual_item := item[:-1]) in self._result_fields:
            return self._agg_result(actual_item)

        result_field_names = [f + "_" for f in self._result_fields]

        raise AttributeError(
            f"Attribute {item} is not a valid attribute for {self.__class__.__name__} nor a dynamically generated "
            "result attribute of the result dataclass. "
            f"Valid result attributes are: {result_field_names}. "
            "Note the trailing underscore!"
        )


class TypedIterator(BaseTypedIterator[DataclassT], Generic[DataclassT]):
    """Helper to iterate over data and collect results.

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

    """

    def iterate(self, iterable: Iterable[T]) -> Iterator[tuple[T, DataclassT]]:
        """Iterate over the given iterable and yield the input and a new empty result object for each iteration.

        Parameters
        ----------
        iterable
            The iterable to iterate over.

        Yields
        ------
        input, result_object
            The input and a new empty result object.
            The result object is a dataclass instance of the type defined in ``self.data_type``.
            All values of the result object are set to ``TypedIterator.NULL_VALUE`` by default.

        """
        yield from self._iterate(iterable)
