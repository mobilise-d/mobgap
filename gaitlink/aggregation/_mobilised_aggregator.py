import typing
import warnings

import numpy as np
import pandas as pd
from typing_extensions import Self

from gaitlink.aggregation.base import BaseAggregator, base_aggregator_docfiller

INPUT_COLUMNS = [
    "duration",
    "step_number",
    "turn_number",
    "stride_speed",
    "stride_length",
    "cadence",
    "stride_duration",
]


# TODO: move these functions somewhere else?
def _custom_quantile(x: float) -> float:
    """Calculate the 90th percentile of the passed data."""
    return np.nanpercentile(x, 90)


def _coefficient_of_variation(x: float) -> float:
    """Calculate variation of the passed data."""
    return x.std() / x.mean()


@base_aggregator_docfiller
class MobilisedAggregator(BaseAggregator):
    """Todo: docs go here."""

    _ALL_WB_AGGS: typing.ClassVar = {
        "walkdur_all_sum": ("duration", "sum"),
        "steps_all_sum": ("step_number", "sum"),
        "turns_all_sum": ("turn_number", "sum"),
        "wb_all_sum": ("duration", "count"),
        "wbdur_all_avg": ("duration", "median"),
        "wbdur_all_max": ("duration", _custom_quantile),
        "wbdur_all_var": ("duration", _coefficient_of_variation),
        "cadence_all_avg": ("cadence", "mean"),
        "strdur_all_avg": ("stride_duration", "mean"),
        "cadence_all_var": ("cadence", _coefficient_of_variation),
        "strdur_all_var": ("stride_duration", _coefficient_of_variation),
    }

    _TEN_THIRTY_WB_AGGS: typing.ClassVar = {
        "ws_1030_avg": ("stride_speed", "mean"),
        "strlen_1030_avg": ("stride_length", "mean"),
    }

    _TEN_WB_AGGS: typing.ClassVar = {
        "wb_10_sum": ("duration", "count"),
        "ws_10_max": ("stride_speed", _custom_quantile),
    }

    _THIRTY_WB_AGGS: typing.ClassVar = {
        "wb_30_sum": ("duration", "count"),
        "ws_30_avg": ("stride_speed", "mean"),
        "strlen_30_avg": ("stride_length", "mean"),
        "cadence_30_avg": ("cadence", "mean"),
        "strdur_30_avg": ("stride_duration", "mean"),
        "ws_30_max": ("stride_speed", _custom_quantile),
        "cadence_30_max": ("cadence", _custom_quantile),
        "ws_30_var": ("stride_speed", _coefficient_of_variation),
        "strlen_30_var": ("stride_length", _coefficient_of_variation),
    }

    _SIXTY_WB_AGGS: typing.ClassVar = {"wb_60_sum": ("duration", "count")}

    _FILTERS_AND_AGGS: typing.ClassVar = [
        (None, _ALL_WB_AGGS),
        ("duration > 10 & duration <= 30", _TEN_THIRTY_WB_AGGS),
        ("duration > 10", _TEN_WB_AGGS),
        ("duration > 30", _THIRTY_WB_AGGS),
        ("duration > 60", _SIXTY_WB_AGGS),
    ]

    _UNIT_CONVERSIONS: typing.ClassVar = {
        "walkdur_all_sum": 1 / 3600,
        "strlen_1030_avg_d": 100,
        "strlen_30_avg_d": 100,
        "strlen_30_var_d": 100,
    }

    _COUNT_COLUMNS: typing.ClassVar = [
        "wb_10_sum_d",
        "wb_30_sum_d",
        "wb_60_sum_d",
        "wb_all_sum_d",
        "steps_all_sum_d",
        "turns_all_sum_d",
    ]

    _ROUND_COLUMNS: typing.ClassVar = [
        "walkdur_all_sum_d",
    ]

    def __init__(self) -> None:
        self.filtered_data = None
        super().__init__()

    def aggregate(
        self,
        data: pd.DataFrame,
        *,
        data_mask: pd.DataFrame,
        groupby_columns: list[str] = ["subject_code", "visit_date"],
    ) -> Self:
        """%(aggregate_short)s.

        Parameters
        ----------
        %(aggregate_para)s

        %(aggregate_return)s
        """
        self.data = data
        self.data_mask = data_mask
        self.groupby_columns = groupby_columns
        self.filtered_data = self.data.copy()

        if not any(col in self.data.columns for col in INPUT_COLUMNS):
            raise ValueError(f"None of the valid input columns {INPUT_COLUMNS} found in the passed dataframe.")

        if not all(col in self.data.reset_index().columns for col in self.groupby_columns):
            raise ValueError(f"Not all groupby columns {self.groupby_columns} found in the passed dataframe.")

        if data_mask is not None:
            if self.data.shape[0] != self.data_mask.shape[0]:
                raise ValueError("The passed data and data_mask do not have the same number of rows.")

            for col in INPUT_COLUMNS:
                flag_col = f"{col}_flag"
                # set entries flagged as implausible to NaN
                if all([col in self.data.columns, flag_col in self.data_mask.columns]):
                    self.filtered_data = self._apply_data_mask_to_col(col, flag_col)
                # as last filtering step, delete all rows with implausible duration
                if col == "duration":
                    self.filtered_data = self._apply_duration_mask(self.filtered_data)

        available_filters_and_aggs = self._select_aggregations()
        self.aggregated_data_ = self._apply_aggregations(available_filters_and_aggs)
        self.aggregated_data_ = self._fillna_count_columns(self.aggregated_data_)
        self.aggregated_data_ = self._convert_units(self.aggregated_data_)
        self.aggregated_data_ = self.aggregated_data_.round(3)

        return self

    def _apply_data_mask_to_col(self, col: str, flag_col: str) -> pd.DataFrame:
        if col in ["cadence", "stride_length"]:
            self.filtered_data.loc[~self.data_mask[flag_col], "stride_speed"] = pd.NA
        self.filtered_data.loc[~self.data_mask[flag_col], col] = pd.NA
        return self.filtered_data

    @staticmethod
    def _apply_duration_mask(data: pd.DataFrame) -> pd.DataFrame:
        return data.dropna(subset=["duration"])

    def _select_aggregations(self) -> list[tuple[str, dict[str, tuple[str, str]]]]:
        """Build list of filters and aggregations to apply.

        Available aggregations are selected based on the columns available in
        the input data.
        """
        available_filters_and_aggs = []
        for filt, aggs in self._FILTERS_AND_AGGS:
            if all([filt is not None, "duration" not in self.data.columns]):
                warnings.warn(
                    f"Filter '{filt}' for walking bout length cannot be applied, "
                    "because the data does not contain a 'duration' column.",
                    stacklevel=2,
                )
                continue

            # check if the property to aggregate is contained in data columns
            available_aggs = {key: value for key, value in aggs.items() if value[0] in self.data.columns}
            if available_aggs:
                available_filters_and_aggs.append((filt, available_aggs))
        return available_filters_and_aggs

    def _apply_aggregations(
        self, available_filters_and_aggs: list[tuple[str, dict[str, tuple[str, str]]]]
    ) -> pd.DataFrame:
        """Apply filters and aggregations to the data."""
        aggregated_results = []
        for f, agg in available_filters_and_aggs:
            aggregated_data = self.filtered_data.copy() if f is None else self.filtered_data.query(f).copy()
            aggregated_results.append(aggregated_data.groupby(self.groupby_columns).agg(**agg))
        return pd.concat(aggregated_results, axis=1)

    @staticmethod
    def _fillna_count_columns(data: pd.DataFrame) -> pd.DataFrame:
        """Replace "NaN" values with 0.

        All "count" parameters are NaN, when no walking bouts of the respective duration are available in a group.
        We replace them with 0.
        """
        count_columns = [col for col in MobilisedAggregator._COUNT_COLUMNS if col in data.columns]
        data.loc[:, count_columns] = data.loc[:, count_columns].fillna(0)
        return data.astype({c: "Int64" for c in count_columns})

    @staticmethod
    def _convert_units(data: pd.DataFrame) -> pd.DataFrame:
        """Convert the units of the aggregated data to the desired output units."""
        for col, factor in MobilisedAggregator._UNIT_CONVERSIONS.items():
            if col in data.columns:
                data.loc[:, col] *= factor
        return data
