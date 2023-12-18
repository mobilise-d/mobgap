import typing
import warnings

import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from gaitlink.aggregation.base import BaseAggregator, base_aggregator_docfiller


def _custom_quantile(x: pd.Series) -> float:
    """Calculate the 90th percentile of the passed data."""
    return np.nanpercentile(x, 90)


def _coefficient_of_variation(x: pd.Series) -> float:
    """Calculate variation of the passed data."""
    return x.std() / x.mean()


@base_aggregator_docfiller
class MobilisedAggregator(BaseAggregator):
    """Implementation of the aggregation algorithm utilized in Mobilise-D.

    This algorithm aggregates DMO parameters across single walking bouts. The calculated parameters are divided into 4
    groups based on the duration of the walking bout. For every group, a different set of possible parameters is
    available.
    The groups are defined as follows:

    - All walking bout [parameters with "all"], available parameters:

      - Duration of all walking bouts in h ["walkdur_all_sum"]
      - Number of steps in all walking bouts ["steps_all_sum"]
      - Number of turns in all walking bouts ["turns_all_sum"]
      - Number of walking bouts ["wb_all_sum"]
      - Median duration of walking bouts in s ["wbdur_all_avg"]
      - 90th percetile of duration of walking bouts in s ["wbdur_all_max"]
      - Coefficient of variation in walking bout duration in s ["wbdur_all_var"]
      - Average cadence of walking bouts in steps/min ["cadence_all_avg"]
      - Average stride duration of walking bouts in s ["strdur_all_avg"]
      - Coefficient of variation in cadence of walking bouts in steps/min ["cadence_all_var"]
      - Coefficient of variation in stride duration of walking bouts in s ["strdur_all_var"]

    - Walking bouts with duration between 10 and 30 seconds [parameters with "1030"], available parameters:

      - Average stride speed of walking bouts in m/s ["ws_1030_avg"]
      - Average stride length of walking bouts in cm ["strlen_1030_avg"]

    - Walking bouts with duration longer than 10 seconds [parameters with "10"], available parameters:

      - Number of walking bouts ["wb_10_sum"]
      - 90th percentile of stride speed of walking bouts in m/s ["ws_10_max"]

    - Walking bouts with duration longer than 30 seconds [parameters with "30"], available parameters:

      - Number of walking bouts ["wb_30_sum"]
      - Average stride speed of walking bouts in m/s ["ws_30_avg"]
      - Average stride length of walking bouts in cm ["strlen_30_avg"]
      - Average cadence of walking bouts in steps/min ["cadence_30_avg"]
      - Average stride duration of walking bouts in s ["strdur_30_avg"]
      - 90th percentile of stride speed of walking bouts in m/s ["ws_30_max"]
      - 90th percentile of cadence of walking bouts in steeps/min ["cadence_30_max"]
      - Coefficient of variation in stride speed of walking bouts in m/s ["ws_30_var"]
      - Coefficient of variation in stride length of walking bouts in cm ["strlen_30_var"]

    - Walking bouts with duration longer than 60 seconds [parameters with "60"], available parameters:

      - Number of walking bouts ["wb_60_sum"]

    Every of the above mentioned parameters will be added to a distinct column in the aggregated data.
    Which of the parameters are calculated depends on the columns available in the input data. All parameters are
    calculated when all of the following columns are available:

    - ``duration_s``
    - ``n_steps``
    - ``n_turns``
    - ``walking_speed_mps``
    - ``stride_length_m``
    - ``cadence_spm``
    - ``stride_duration_s``

    Otherwise, only parameters for which the corresponding DMO data is provided are added to the aggregation results.
    For example, if the input data does not contain a "stride_length" column, the "strlen_1030_avg", "strlen_30_avg",
    "strlen_30_var" parameters are not calculated. Furthermore, if no "duration" is provided, only the "all"-parameters
    without duration filter will be calculated.

    The aggregation parameters are calculated for every unique group of the ``groupby_columns``. Per default,
    one set of aggregation results is calculated per participant and recording date. This can however be adapted by
    passing a different list of ``groupby_columns``.

    Parameters
    ----------
    groupby_columns
        A list of columns to group the data by. Based on the resulting groups, the aggregations are calculated.
        Possible groupings are e.g. by participant, recording date, or trial.
        To generate daily aggregations (the default), the ``groupby_columns`` should contain the columns "subject_code"
        and "visit_date".

    Other Parameters
    ----------------
    %(other_parameters)s
    data_mask
        A boolean DataFrame with the same number of rows as ``data`` indicating the validity of every measure. Every
        column of the data mask corresponds to a column of ``data`` and has the same name.
        If an entry is ``False``, the corresponding measure is implausible and should be ignored for the aggregations.

        To exclude implausible data points from the input data, a ``data_mask`` can be passed to the ``aggregate``
        method.
        The columns in data_mask are regarded if there exists a column in the input data with the same name.
        Note that depending on which DMO measure is flagged as implausible, different elimination steps are applied:

        - "duration_s": The whole walking bout is not considered for the aggregation.
        - "n_steps": The corresponding "n_steps" is not regarded.
        - "n_turns": The corresponding "turn_number" is not regarded.
        - "walking_speed_mps": The corresponding "stride_speed" is not regarded.
        - "stride_length_m": The corresponding "stride_length" AND the corresponding "stride_speed" are not regarded.
        - "cadence_spm": The corresponding "cadence" AND the corresponding "stride_speed" are not regarded.
        - "stride_duration_s": The corresponding "stride_duration" is not regarded.

    Attributes
    ----------
    %(aggregated_data_)s
    filtered_data_
        An updated version of ``data`` with the implausible entries removed based on ``data_mask``.
        Depending on the implementation, the shape of ``filtered_data_`` might differ from ``data``.

    Notes
    -----
    The outputs of this aggregation algorithm are analogous to the outputs of the original Mobilise-D R-Script for
    aggregation.
    However, there can be small differences in the second/third decimal place range in the results. This is due to
    different outputs of the quantile function in Python and R.
    Furthermore, the parameter "strlen_30_var" is converted to cm for consistency, while it is in m in the original
    R-Script.
    """

    INPUT_COLUMNS: typing.ClassVar[list[str]] = [
        "stride_duration_s",
        "n_steps",
        "n_turns",
        "walking_speed_mps",
        "stride_length_m",
        "cadence_spm",
        "stride_duration_s",
    ]

    _ALL_WB_AGGS: typing.ClassVar[dict[str, tuple[str, typing.Union[str, typing.Callable]]]] = {
        "walkdur_all_sum": ("duration_s", "sum"),
        "steps_all_sum": ("n_steps", "sum"),
        "turns_all_sum": ("n_turns", "sum"),
        "wb_all_sum": ("duration_s", "count"),
        "wbdur_all_avg": ("duration_s", "median"),
        "wbdur_all_max": ("duration_s", _custom_quantile),
        "wbdur_all_var": ("duration_s", _coefficient_of_variation),
        "cadence_all_avg": ("cadence_spm", "mean"),
        "strdur_all_avg": ("stride_duration_s", "mean"),
        "cadence_all_var": ("cadence_spm", _coefficient_of_variation),
        "strdur_all_var": ("stride_duration_s", _coefficient_of_variation),
    }

    _TEN_THIRTY_WB_AGGS: typing.ClassVar = {
        "ws_1030_avg": ("walking_speed_mps", "mean"),
        "strlen_1030_avg": ("stride_length_m", "mean"),
    }

    _TEN_WB_AGGS: typing.ClassVar = {
        "wb_10_sum": ("duration_s", "count"),
        "ws_10_max": ("walking_speed_mps", _custom_quantile),
    }

    _THIRTY_WB_AGGS: typing.ClassVar = {
        "wb_30_sum": ("duration_s", "count"),
        "ws_30_avg": ("walking_speed_mps", "mean"),
        "strlen_30_avg": ("stride_length_m", "mean"),
        "cadence_30_avg": ("cadence_spm", "mean"),
        "strdur_30_avg": ("stride_duration_s", "mean"),
        "ws_30_max": ("walking_speed_mps", _custom_quantile),
        "cadence_30_max": ("cadence_spm", _custom_quantile),
        "ws_30_var": ("walking_speed_mps", _coefficient_of_variation),
        "strlen_30_var": ("stride_length_m", _coefficient_of_variation),
    }

    _SIXTY_WB_AGGS: typing.ClassVar = {"wb_60_sum": ("duration_s", "count")}

    _FILTERS_AND_AGGS: typing.ClassVar = [
        (None, _ALL_WB_AGGS),
        ("duration_s > 10 & duration_s <= 30", _TEN_THIRTY_WB_AGGS),
        ("duration_s > 10", _TEN_WB_AGGS),
        ("duration_s > 30", _THIRTY_WB_AGGS),
        ("duration_s > 60", _SIXTY_WB_AGGS),
    ]

    _UNIT_CONVERSIONS: typing.ClassVar = {
        "walkdur_all_sum": 1 / 3600,
        "strlen_1030_avg": 100,
        "strlen_30_avg": 100,
        "strlen_30_var": 100,
    }

    _COUNT_COLUMNS: typing.ClassVar = [
        "wb_10_sum",
        "wb_30_sum",
        "wb_60_sum",
        "wb_all_sum",
        "steps_all_sum",
        "turns_all_sum",
    ]

    _ROUND_COLUMNS: typing.ClassVar = [
        "walkdur_all_sum",
    ]

    data_mask: pd.DataFrame
    filtered_data_: pd.DataFrame

    def __init__(self, groupby_columns: typing.Sequence[str] = ("visit_type", "participant_id", "measurement_date")) -> None:
        self.groupby_columns = groupby_columns

    @base_aggregator_docfiller
    def aggregate(
        self,
        data: pd.DataFrame,
        data_mask: typing.Union[pd.DataFrame, None] = None,
        **_: Unpack[dict[str, typing.Any]],
    ) -> Self:
        """%(aggregate_short)s.

        Parameters
        ----------
        %(aggregate_para)s

        %(aggregate_return)s
        """
        self.data = data
        self.data_mask = data_mask
        self.filtered_data_ = self.data.copy()
        groupby_columns = list(self.groupby_columns)

        # TODO: Align index of data an mask to ensure that all rows exist in both dataframes and the correct rows are
        #      removed in the filtering step

        if not any(col in self.data.columns for col in self.INPUT_COLUMNS):
            raise ValueError(f"None of the valid input columns {self.INPUT_COLUMNS} found in the passed dataframe.")

        if not all(col in self.data.reset_index().columns for col in groupby_columns):
            raise ValueError(f"Not all groupby columns {self.groupby_columns} found in the passed dataframe.")

        if data_mask is not None:
            try:
                data_mask = data_mask.loc[self.data.index]
            except KeyError as e:
                raise ValueError(
                    f"The datamask seems to be missing some data indices. "
                    "The datamask must have exactly the same indices as the data."
                ) from e

            if self.data.shape[0] != self.data_mask.shape[0]:
                raise ValueError("The passed data and data_mask do not have the same number of rows.")

            for col in self.INPUT_COLUMNS:
                # remove walking bouts in the last filtering step
                if col == "duration_s":
                    continue
                # set entries flagged as implausible to NaN
                if all([col in self.data.columns, col in data_mask.columns]):
                    self.filtered_data_ = self._apply_data_mask_to_col(self.filtered_data_, data_mask, col)

            # as last filtering step, delete all rows with implausible duration
            if all(["duration_s" in self.data.columns, "duration_s" in self.data_mask.columns]):
                self.filtered_data_ = self._apply_duration_mask(self.filtered_data_, self.data_mask["duration_s"])

        available_filters_and_aggs = self._select_aggregations(self.data.columns)
        self.aggregated_data_ = self._apply_aggregations(
            self.filtered_data_, groupby_columns, available_filters_and_aggs
        )
        self.aggregated_data_ = self._fillna_count_columns(self.aggregated_data_)
        self.aggregated_data_ = self._convert_units(self.aggregated_data_)
        self.aggregated_data_ = self.aggregated_data_.round(3)

        return self

    @staticmethod
    def _apply_data_mask_to_col(data: pd.DataFrame, mask_col: pd.Series, col: str) -> pd.DataFrame:
        """Clean the data according to a column of the data mask."""
        if col == "cadence_spm":
            # If cadence is implausible, stride speed is also implausible
            data.loc[~mask_col[col], "stride_length_m"] = pd.NA
        data.loc[~mask_col[col], col] = pd.NA
        return data

    @staticmethod
    def _apply_duration_mask(data: pd.DataFrame, duration_mask: pd.Series) -> pd.DataFrame:
        """Remove rows with implausible duration."""
        return data.loc[duration_mask.index]

    def _select_aggregations(
        self, data_columns: list[str]
    ) -> list[tuple[str, dict[str, tuple[str, typing.Union[str, typing.Callable]]]]]:
        """Build list of filters and aggregations to apply.

        Available aggregations are selected based on the columns available in
        the input data.
        """
        available_filters_and_aggs = []
        for filt, aggs in self._FILTERS_AND_AGGS:
            if all([filt is not None, "duration_s" not in data_columns]):
                warnings.warn(
                    f"Filter '{filt}' for walking bout length cannot be applied, "
                    "because the data does not contain a 'duration' column.",
                    stacklevel=2,
                )
                continue

            # check if the property to aggregate is contained in data columns
            available_aggs = {key: value for key, value in aggs.items() if value[0] in data_columns}
            if available_aggs:
                available_filters_and_aggs.append((filt, available_aggs))
        return available_filters_and_aggs

    @staticmethod
    def _apply_aggregations(
        filtered_data: pd.DataFrame,
        groupby_columns: list[str],
        available_filters_and_aggs: list[tuple[str, dict[str, tuple[str, typing.Union[str, typing.Callable]]]]],
    ) -> pd.DataFrame:
        """Apply filters and aggregations to the data."""
        aggregated_results = []
        for f, agg in available_filters_and_aggs:
            aggregated_data = filtered_data.copy() if f is None else filtered_data.query(f).copy()
            aggregated_results.append(aggregated_data.groupby(groupby_columns).agg(**agg))
        return pd.concat(aggregated_results, axis=1)

    def _fillna_count_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replace "NaN" values with 0.

        All "count" parameters are NaN, when no walking bouts of the respective duration are available in a group.
        We replace them with 0.
        """
        count_columns = [col for col in self._COUNT_COLUMNS if col in data.columns]
        data.loc[:, count_columns] = data.loc[:, count_columns].fillna(0)
        return data.astype({c: "Int64" for c in count_columns})

    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert the units of the aggregated data to the desired output units."""
        for col, factor in self._UNIT_CONVERSIONS.items():
            if col in data.columns:
                data.loc[:, col] *= factor
        return data
