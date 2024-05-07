import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Generic, NamedTuple, Optional, TypeVar

from tpcp import Algorithm, cf

DataclassT = TypeVar("DataclassT")
T = TypeVar("T")


class _NotSet:
    def __repr__(self):
        return "_NOT_SET"


InputTypeT = TypeVar("InputTypeT")
ResultT = TypeVar("ResultT")


class TypedIteratorResultTuple(NamedTuple, Generic[InputTypeT, ResultT]):
    iteration_name: str
    input: InputTypeT
    result: ResultT
    iteration_context: dict[str, Any]


class BaseTypedIterator(Algorithm, Generic[InputTypeT, DataclassT]):
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
    aggregations: Sequence[tuple[str, Callable[[TypedIteratorResultTuple[InputTypeT, Generic[ResultT]]], Any]]]

    _raw_results: list[TypedIteratorResultTuple[InputTypeT, DataclassT]]
    _result_fields: set[str]
    _raw_input_context: Any
    done_: dict[str, bool]

    NULL_VALUE = _NotSet()

    def __init__(
        self,
        data_type: type[DataclassT],
        aggregations: Sequence[
            tuple[str, Callable[[TypedIteratorResultTuple[InputTypeT, Generic[ResultT]]], Any]]
        ] = cf([]),
    ):
        self.data_type = data_type
        self.aggregations = aggregations

    def _iterate(
        self,
        iterable: Iterable[tuple[T]],
        *,
        iteration_name: str = "__main__",
        iteration_context: Optional[dict[str, Any]] = None,
    ) -> Iterator[tuple[T, DataclassT]]:
        """Iterate over the given iterable and yield the input and a new empty result object for each iteration.

        Parameters
        ----------
        iterable
            The iterable to iterate over.
            Note, that the iterable is expected to yield a tuple of the form `(unique_iteration_id, value)` for each
            iteration.
            The iteration id is used internally to keep track of what results belong to which iteration.
        iteration_name
            The name of the iteration.
            This is an advanced feature and should only be used if you want to have multiple nested iterations.
            See the iterator example for potential usecases.
        iteration_context
            This is any piece of information that you want to pass to all aggregation functions via the result object.

        Yields
        ------
        input, result_object
            The input and a new empty result object.
            The result object is a dataclass instance of the type defined in ``self.data_type``.
            All values of the result object are set to ``TypedIterator.NULL_VALUE`` by default.

        """
        if not is_dataclass(self.data_type):
            raise TypeError(f"Expected a dataclass as data_type, got {self.data_type}")

        if iteration_name == "__main__":
            self.done_ = {}

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

        self.done_[iteration_name] = False
        for d in iterable:
            result_object = self._get_new_empty_object()
            result_tuple = TypedIteratorResultTuple(iteration_name, d, result_object, iteration_context or {})
            self._report_new_result(result_tuple)
            yield d, result_object
        self.done_[iteration_name] = True

    def _get_new_empty_object(self) -> DataclassT:
        init_dict = {k.name: self.NULL_VALUE for k in fields(self.data_type)}
        return self.data_type(**init_dict)

    def _report_new_result(self, r: TypedIteratorResultTuple[InputTypeT, DataclassT]):
        self._raw_results.append(r)

    @property
    def raw_results_(self) -> list[TypedIteratorResultTuple[InputTypeT, Any]]:
        if "__main__" not in self.done_:
            raise ValueError(
                "The iterator has not been started yet. No results are available. Call the iterate method first."
            )
        if not self.done_["__main__"]:
            warnings.warn("The iterator is not done yet. The results might not be complete.", stacklevel=1)

        # We invert the results. For each result we check if a result attribute is set and turn all results in a list
        # for each attribute.
        inverted_results = {}
        for r in self._raw_results:
            for a in fields(self.data_type):
                result_copy = r._asdict()
                result_copy["result"] = getattr(r.result, a.name)
                inverted_results.setdefault(a.name, []).append(TypedIteratorResultTuple(**result_copy))

        return self.data_type(**inverted_results)

    def _agg_result(self, raw_results: DataclassT):
        aggregations = dict(self.aggregations)
        agg_results = {}
        for a in fields(self.data_type):
            # if an aggregator is defined for the specific item, we apply it
            name = a.name
            values = getattr(raw_results, name)
            if name in aggregations:
                values = aggregations[name](values)
            agg_results[name] = values
        return agg_results

    @property
    def results_(self) -> DataclassT:
        """The aggregated results.

        Note, that this returns an instance of the result object, even-though the datatypes of the attributes might be
        different depending on the aggregation function.
        We still decided it makes sense to return an instance of the result object, as it will allow to autocomplete
        the attributes, even-though the associated times might not be correct.
        """
        return self.data_type(**self._agg_result(self.raw_results_))


class TypedIterator(BaseTypedIterator[InputTypeT, DataclassT], Generic[InputTypeT, DataclassT]):
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
