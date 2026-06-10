import typing
import warnings
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from pandas import option_context
from tpcp import cf
from tpcp.misc import set_defaults
from typing_extensions import Self, Unpack

from mobgap.aggregation.base import BaseAggregator, base_aggregator_docfiller


def _custom_quantile_90(x: pd.Series) -> float:
    """Calculate the 90th percentile of the passed data."""
    if x.isna().all():
        return np.nan
    return np.nanpercentile(x, 90)

def _custom_quantile_10(x: pd.Series) -> float:
    """Calculate the 90th percentile of the passed data."""
    if x.isna().all():
        return np.nan
    return np.nanpercentile(x, 10)


class SDMOAggregator(BaseAggregator):
    """Implementation of the aggregations used for the signal-based digital mobility outcomes.

    This aggregator automatically detects all columns in the input DataFrame (except `duration_s` and `groupby` columns)
    and computes a predefined set of statistics (mean, 90th/10th percentile, coefficient of variation) for each of the
    default filter groups (all WBs, WBs between 10s and 30s and WBs greater than 10s, 30s, 60s).

    Depending on available columns in the input data, calculated parameters will be added to a distinct column
    in the aggregated data.

    If no ``duration_filters`` is provided, only the "all"-parameters without duration filter will be calculated.

    Similar to the ``MobilisedAggregator``, the aggregation parameters are calculated for every unique group of the
    ``groupby``. Per default, one set of aggregation results is calculated per participant and recording date.
    This can however be adapted by passing a different list of ``groupby``.

    Here, due to the large number of SDMOs, original names are not provided as in the primary DMOs in
    ``MobilisedAggregator``, but, the same naming convention is followed. For example, ``all__RMS_acc_is__p90``
    refers to the 90th percentile of root mean square acceleration (RMS_acc) in the  vertical (``is``) in all WBs.


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
        Aka ``wb_sdmos.reset_index().set_index([*groupby, unique_wb_id_column]).index.is_unique`` must be ``True``.

    Other Parameters
    ----------------
    %(other_parameters)s
    wb_sdmos_mask
        A boolean DataFrame with the same shape the ``wb_sdmos`` indicating the validity of every measure.
        See ``MobilisedAggregator`` for more details.
        For the signal-based DMOs, it is unlikely that this parameter will be set, so it defaults to None, meaning
        that no masking will be performed. However, to align this with the main aggregator and allow flexibility,
        this parameter is defined.

    Attributes
    ----------
    %(aggregated_data_)s
    filtered_wb_sdmos_
        An updated version of ``wb_sdmos`` with the implausible entries removed based on ``wb_sdmos_mask``.
        ``filtered_wb_sdmos_`` will have the groupby columns and the ``unique_wb_id_column`` set as index.
    """

    groupby: typing.Optional[typing.Sequence[str]]
    wb_sdmos_mask: pd.DataFrame
    filtered_wb_sdmos_: pd.DataFrame

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
        duration_filters: typing.Optional[dict[str, tuple[float, float]]],
        metrics: typing.Optional[dict[str, typing.Callable]],
        unique_wb_id_column: str,
    ) -> None:
        self.groupby = groupby
        self.duration_filters = duration_filters
        self.metrics = metrics
        self.unique_wb_id_column = unique_wb_id_column

    @base_aggregator_docfiller
    def aggregate(  # noqa: C901, PLR0912
        self,
        wb_sdmos: pd.DataFrame,
        *,
        wb_sdmos_mask: typing.Union[pd.DataFrame, None] = None,
        **_: Unpack[dict[str, typing.Any]],
    ) -> Self:
        """%(aggregate_short)s.

        Parameters
        ----------
        %(aggregate_para)s
        wb_sdmos_mask
            A boolean DataFrame with the same shape the ``wb_sdmos`` indicating the validity of every measure.
            If the DataFrame contains a ``NaN`` value, this is interpreted as ``True``, assuming no checks were applied
            to this value and the corresponding measure is regarded as plausible.

        %(aggregate_return)s
        """
        self.wb_sdmos = wb_sdmos
        self.wb_sdmos_mask = wb_sdmos_mask
        groupby = self.groupby if self.groupby is None else list(self.groupby)
        duration_col = "duration_s"

        # select columns to aggregate, except duration_col and groupby
        exclude_cols = {duration_col}
        if groupby:
            # filter any groupby col in the index
            exclude_cols.update([c for c in groupby if c in wb_sdmos.columns])
        value_cols = [c for c in wb_sdmos.columns if c not in exclude_cols]

        if not value_cols:
            warnings.warn(
                f"No valid columns to aggregate found after excluding {exclude_cols}.",
                stacklevel=2,
            )
            self.aggregated_data_ = pd.DataFrame()
            return self

        if groupby and not all(col in wb_sdmos.reset_index().columns for col in groupby):
            raise ValueError(f"Not all groupby columns {groupby} found in the passed dataframe.")

        data_correct_index = wb_sdmos.reset_index().set_index([*(groupby or []), self.unique_wb_id_column]).sort_index()

        if not data_correct_index.index.is_unique:
            raise ValueError(
                f"The passed data contains multiple entries for the same groupby columns {groupby}. "
                "Make sure that the passed data in `unique_wb_id_column` is unique for every groupby "
                "column combination."
            )

        if wb_sdmos_mask is not None:
            wb_sdmos_mask = (
                wb_sdmos_mask.fillna(True)
                .reset_index()
                .set_index([*(groupby or []), self.unique_wb_id_column])
                .sort_index()
                .astype(bool)
            )
            if not data_correct_index.index.equals(wb_sdmos_mask.index):
                raise ValueError(
                    "The data mask seems to be missing some data indices. "
                    "`wb_sdmos_mask` must have exactly the same indices as `wb_sdmos` after grouping."
                )

            # We remove all individual elements from the data that are flagged as implausible in the data mask.
            self.filtered_wb_sdmos_ = data_correct_index.where(wb_sdmos_mask)
            # implausible bout duration
            if duration_col in data_correct_index.columns and duration_col in wb_sdmos_mask.columns:
                # If the duration is implausible, we need to remove the whole walking bout
                self.filtered_wb_sdmos_ = self.filtered_wb_sdmos_.where(wb_sdmos_mask[duration_col])
            # for the other columns (SDMOs)
            cols_to_check = wb_sdmos_mask.columns.intersection(value_cols)
            combined_mask = wb_sdmos_mask[cols_to_check].all(axis=1)
            self.filtered_wb_sdmos_ = self.filtered_wb_sdmos_.loc[combined_mask]
        else:
            self.filtered_wb_sdmos_ = data_correct_index.copy()

        # select filters and aggregations
        # Resulting column names: f"{group_name}__{col}__{metric_name}" following the convention for the primary DMOs
        filters_and_aggs = []
        for group_name, (low, high) in self.duration_filters.items():
            if low == 0 and np.isinf(high):
                query = None
            elif np.isinf(high):
                query = f"{duration_col} >= {low}"
            else:
                query = f"{duration_col} >= {low} & {duration_col} < {high}"
            agg_dict = {}
            for col in value_cols:
                for metric_name, func in self.metrics.items():
                    agg_dict[f"{group_name}__{col}__{metric_name}"] = (col, func)
            filters_and_aggs.append((query, agg_dict))

        # apply aggregations
        aggregated_results = []
        for query, agg in filters_and_aggs:
            internal_filtered = self.filtered_wb_sdmos_
            if query is not None:
                internal_filtered = internal_filtered.query(query)
            if groupby:
                grouped = internal_filtered.groupby(groupby)
            else:
                # "all_wbs" index is assigned to match the aggregated data of the MobilisedAggregator
                grouped = internal_filtered.groupby(pd.Series("all_wbs", index=internal_filtered.index))
            aggregated_results.append(grouped.agg(**agg))
        self.aggregated_data_ = pd.concat(aggregated_results, axis=1)
        return self