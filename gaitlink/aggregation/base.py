"""Base class for aggregators."""
from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from gaitlink._docutils import make_filldoc

base_aggregator_docfiller = make_filldoc(
    {
        # TODO
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseAggregator`.",
)


@base_aggregator_docfiller
class BaseAggregator(Algorithm):
    """
    Base class for aggregators.

    This base class should be used for all aggregation algorithms.
    Algorithms should implement the ``aggregate`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``aggregated_data_`` attribute set to the
    calculated aggregations.

    We allow that subclasses specify further parameters for the detect methods (hence, this baseclass supports
    ``**kwargs``).
    However, you should only use them, if you really need them and apply active checks, that they are passed correctly.
    In 99% of the time, you should add a new parameter to the algorithm itself, instead of adding a new parameter to
    the ``aggregate`` method.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(aggregated_data_)

    Notes
    -----
    You can use the :func:`~base_aggregator_docfiller` decorator to fill common parts of the docstring for your
    subclass. See the source of this class for an example.
    """

    _action_methods = ("aggregate",)

    # Other Parameters
    data: pd.DataFrame
    data_mask: pd.DataFrame
    groupby_columns: list[str]

    # results
    aggregated_data_: pd.DataFrame

    @base_aggregator_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        *,
        data_mask: pd.DataFrame,
        groupby_columns: list[str],
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """%(aggregate_short)s.

        Parameters
        ----------
        %(aggregate_para)s

        %(aggregate_return)s
        """
        raise NotImplementedError


__all__ = ["BaseAggregator", "base_aggregator_docfiller"]
