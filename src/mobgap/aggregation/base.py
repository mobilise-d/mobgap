"""Base class for aggregators."""

from typing import Any

import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap._docutils import make_filldoc

base_aggregator_docfiller = make_filldoc(
    {
        "other_parameters": """
    wb_dmos
        The DMO data per walking bout passed to the ``aggregate`` method.
    """,
        "aggregated_data_": """
    aggregated_data_
        A dataframe containing the aggregated results.
        The index of the dataframe contains the ``groupby_columns``. Consequently, there is one row which
        aggregation results for each group.
    """,
        "aggregate_short": """
    Aggregate parameters across walking bouts.
    """,
        "aggregate_para": """
    wb_dmos
       The DMO data per walking bout.
       This is a dataframe with one row for every walking bout and one column for every DMO parameter.
       This should further have relevant metadata (i.e. ``participant_id``, ``visit_date``, ``wb_id``) as columns or
       indices.
       The specific requirements depend on the aggregation algorithm.
    """,
        "aggregate_return": """
    Returns
    -------
    self
        The instance of the class with the ``aggregated_data_`` attribute set to the aggregation results.
    """,
    },
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseAggregator`.",
)


@base_aggregator_docfiller
class BaseAggregator(Algorithm):
    """Base class for aggregators.

    This base class should be used for all aggregation algorithms.
    Algorithms should implement the ``aggregate`` method, which will perform all relevant processing steps.
    The method should then return the instance of the class, with the ``aggregated_data_`` attribute set to the
    calculated aggregations.

    We allow that subclasses specify further parameters for the aggregate methods (hence, this baseclass supports
    ``**kwargs``).
    However, you should only use them, if you really need them and apply active checks, that they are passed correctly.
    In 99%% of the time, you should add a new parameter to the algorithm itself, instead of adding a new parameter to
    the ``aggregate`` method.

    Other Parameters
    ----------------
    %(other_parameters)s

    Attributes
    ----------
    %(aggregated_data_)s

    Notes
    -----
    You can use the :func:`~base_aggregator_docfiller` decorator to fill common parts of the docstring for your
    subclass. See the source of this class for an example.
    """

    _action_methods = ("aggregate",)

    # Other Parameters
    wb_dmos: pd.DataFrame

    # results
    aggregated_data_: pd.DataFrame

    @base_aggregator_docfiller
    def aggregate(
        self,
        wb_dmos: pd.DataFrame,
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
