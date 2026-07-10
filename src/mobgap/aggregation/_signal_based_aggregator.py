import typing
import warnings
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self

from mobgap.aggregation.base import BaseAggregator, base_aggregator_docfiller


def _custom_quantile_90(x: pd.Series) -> float:
    """Calculate the 90th percentile of the passed data."""
    if x.isna().all():
        return np.nan
    return np.nanpercentile(x, 90)


def _custom_quantile_10(x: pd.Series) -> float:
    """Calculate the 10th percentile of the passed data."""
    if x.isna().all():
        return np.nan
    return np.nanpercentile(x, 10)


@base_aggregator_docfiller
class SDMOAggregator(BaseAggregator):
    """Implementation of the aggregations used for the signal-based digital mobility outcomes.

    This aggregator automatically detects all columns in the input DataFrame, except the duration, walking-bout ID,
    and grouping columns. It computes the median, standard deviation, and 10th and 90th percentiles for each default
    duration group: all WBs, WBs greater than 10, 30, or 60 seconds, and the bounded 10-30 and 30-60 second groups.

    Depending on available columns in the input data, calculated parameters will be added to a distinct column
    in the aggregated data.

    Similar to the ``MobilisedAggregator``, the aggregation parameters are calculated for every unique group of the
    ``groupby``. Per default, one set of aggregation results is calculated per participant and recording date.
    This can however be adapted by passing a different list of ``groupby``.

    Here, due to the large number of SDMOs, original names are not provided as in the primary DMOs in
    ``MobilisedAggregator``, but the same naming convention is followed. For example, ``all__rms_acc_is__p90``
    refers to the 90th percentile of root mean square acceleration in the vertical (``is``) direction across all WBs.


    Parameters
    ----------
    groupby
        A list of columns to group the data by. Based on the resulting groups, the aggregations are calculated.
        Possible groupings are e.g. by participant, recording date, or trial.
        To generate daily aggregations (the default), the ``groupby`` should contain the columns "subject_code"
        and "visit_date".
    duration_filters
        Duration groups. If not provided, uses the default groups.
    metrics
        Definition of metrics to compute for each column. If not provided, uses the defaults provided in
        `PredefinedParameters`.
    unique_wb_id_column
        The name of the column (or index level) containing a unique identifier for every walking bout.
        The id does not have to be unique globally, but only within the groups defined by ``groupby``.
        Aka ``wb_dmos.reset_index().set_index([*groupby, unique_wb_id_column]).index.is_unique`` must be ``True``.

    Other Parameters
    ----------------
    wb_dmos
        The SDMO data per walking bout passed to the ``aggregate`` method.
    wb_dmos_mask
        A boolean DataFrame with the same shape the ``wb_dmos`` indicating the validity of every measure.
        See ``MobilisedAggregator`` for more details.
        For the signal-based DMOs, it is unlikely that this parameter will be set, so it defaults to None, meaning
        that no masking will be performed. However, to align this with the main aggregator and allow flexibility,
        this parameter is defined.

    Attributes
    ----------
    aggregated_data_
        A dataframe containing the aggregated results.
        The index of the dataframe contains the ``groupby_columns``. Consequently, there is one row which
        aggregation results for each group.
    filtered_wb_dmos_
        An updated version of ``wb_dmos`` with implausible entries removed based on ``wb_dmos_mask``.
        ``filtered_wb_dmos_`` will have the groupby columns and the ``unique_wb_id_column`` set as index.
    """

    groupby: typing.Optional[typing.Sequence[str]]
    unique_wb_id_column: str
    wb_dmos_mask: pd.DataFrame
    filtered_wb_dmos_: pd.DataFrame

    # Other Parameters
    wb_dmos: pd.DataFrame

    class PredefinedParameters:
        cvs_sdmo_data: Final = MappingProxyType(
            {
                "groupby": ["visit_type", "participant_id", "measurement_date"],
                "unique_wb_id_column": "wb_id",
                "duration_filters": {
                    "all": (0, np.inf),
                    "wb_10_30": (10, 30),
                    "wb_10": (10, np.inf),
                    "wb_30": (30, np.inf),
                    "wb_30_60": (30, 60),
                    "wb_60": (60, np.inf),
                },
                "metrics": {
                    "median": "median",
                    "std": "std",
                    "p10": _custom_quantile_10,
                    "p90": _custom_quantile_90,
                },
            }
        )

        single_recording: Final = MappingProxyType(
            {
                "groupby": None,
                "unique_wb_id_column": "wb_id",
                "duration_filters": cvs_sdmo_data["duration_filters"],
                "metrics": cvs_sdmo_data["metrics"],
            }
        )

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.single_recording.items()})
    def __init__(
        self,
        groupby: typing.Optional[typing.Sequence[str]],
        *,
        duration_filters: dict[str, tuple[float, float]],
        metrics: dict[str, typing.Callable | str],
        unique_wb_id_column: str,
    ) -> None:
        self.groupby = groupby
        self.duration_filters = duration_filters
        self.metrics = metrics
        self.unique_wb_id_column = unique_wb_id_column

    @base_aggregator_docfiller
    def aggregate(
        self,
        wb_dmos: pd.DataFrame,
        *,
        wb_dmos_mask: pd.DataFrame | None = None,
    ) -> Self:
        """%(aggregate_short)s.

        Parameters
        ----------
        %(aggregate_para)s
        wb_dmos_mask
            A boolean DataFrame with the same shape the ``wb_dmos`` indicating the validity of every measure.
            If the DataFrame contains a ``NaN`` value, this is interpreted as ``True``, assuming no checks were applied
            to this value and the corresponding measure is regarded as plausible.

        %(aggregate_return)s
        """
        self.wb_dmos = wb_dmos
        self.wb_dmos_mask = wb_dmos_mask
        groupby = self.groupby if self.groupby is None else list(self.groupby)
        duration_col = "duration_s"

        exclude_cols = {duration_col, self.unique_wb_id_column}
        exclude_cols.update(groupby or [])
        value_cols = [c for c in wb_dmos.columns if c not in exclude_cols]

        if not value_cols:
            warnings.warn(
                "No valid columns to aggregate found in the passed dataframe.",
                stacklevel=2,
            )
            self.aggregated_data_ = pd.DataFrame()
            return self

        if groupby and not all(col in wb_dmos.reset_index().columns for col in groupby):
            raise ValueError(f"Not all groupby columns {groupby} found in the passed dataframe.")

        data_correct_index = wb_dmos.reset_index().set_index([*(groupby or []), self.unique_wb_id_column]).sort_index()

        if not data_correct_index.index.is_unique:
            raise ValueError(
                f"The passed data contains multiple entries for the same groupby columns {groupby}. "
                "Make sure that the passed data in `unique_wb_id_column` is unique for every groupby "
                "column combination."
            )

        self.filtered_wb_dmos_ = data_correct_index.copy()
        if wb_dmos_mask is not None:
            indexed_mask = (
                wb_dmos_mask.fillna(True)
                .reset_index()
                .set_index([*(groupby or []), self.unique_wb_id_column])
                .sort_index()
                .astype(bool)
            )
            if not data_correct_index.index.equals(indexed_mask.index):
                raise ValueError(
                    "The data mask seems to be missing some data indices. "
                    "`wb_dmos_mask` must have exactly the same indices as `wb_dmos` after grouping."
                )
            if not data_correct_index.columns.equals(indexed_mask.columns):
                raise ValueError("`wb_dmos_mask` must have exactly the same columns as `wb_dmos` after grouping.")

            self.filtered_wb_dmos_ = data_correct_index.where(indexed_mask)
            if duration_col in data_correct_index.columns and duration_col in indexed_mask.columns:
                self.filtered_wb_dmos_.loc[~indexed_mask[duration_col], :] = np.nan

        # select filters and aggregations
        # Resulting column names: f"{group_name}__{col}__{metric_name}" following the convention for the primary DMOs
        filters_and_aggs = self._select_filters_and_aggregations(value_cols)

        # apply aggregations
        aggregated_results = []
        for query, agg in filters_and_aggs:
            internal_filtered = self.filtered_wb_dmos_.query(query) if query is not None else self.filtered_wb_dmos_
            grouping = groupby or pd.Series("all_wbs", index=internal_filtered.index)
            grouped = internal_filtered.groupby(grouping)
            aggregated_results.append(grouped.agg(**agg))
        self.aggregated_data_ = pd.concat(aggregated_results, axis=1)
        return self

    def _select_filters_and_aggregations(self, value_cols: list[str]) -> list[tuple[str, dict]]:
        duration_col = "duration_s"
        duration_filters = self.duration_filters
        if duration_col not in self.wb_dmos.columns and "all" not in self.duration_filters:
            raise ValueError(
                "the data does not contain 'duration_s' column AND no 'all' duration filter (using all walking bouts) "
                "exists. This configuration is ambiguous and cannot be resolved."
            )
        if duration_col not in self.wb_dmos.columns:
            warnings.warn(
                "Filters based on walking bout duration cannot be applied, "
                "because the data does not contain a 'duration_s' column.",
                stacklevel=2,
            )
            duration_filters = {"all": self.duration_filters["all"]}

        filters_and_aggs = []
        for group_name, (low, high) in duration_filters.items():
            if low == 0 and np.isinf(high):
                query = None
            elif np.isinf(high):
                query = f"{duration_col} > {low}"
            else:
                query = f"{duration_col} > {low} & {duration_col} <= {high}"
            agg_dict = {}
            for col in value_cols:
                for metric_name, func in self.metrics.items():
                    agg_dict[f"{group_name}__{col}__{metric_name}"] = (col, func)
            filters_and_aggs.append((query, agg_dict))
        return filters_and_aggs
