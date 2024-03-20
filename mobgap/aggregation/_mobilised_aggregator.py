import typing
import warnings

import numpy as np
import pandas as pd
from typing_extensions import Self, Unpack

from mobgap.aggregation.base import BaseAggregator, base_aggregator_docfiller


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
      - 90th percentile of duration of walking bouts in s ["wbdur_all_max"]
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

    Every of the above-mentioned parameters will be added to a distinct column in the aggregated data.
    Which of the parameters are calculated depends on the columns available in the input data. All parameters are
    calculated when all the following columns are available:

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

    The aggregation parameters are calculated for every unique group of the ``groupby``. Per default,
    one set of aggregation results is calculated per participant and recording date. This can however be adapted by
    passing a different list of ``groupby``.

    Parameters
    ----------
    groupby
        A list of columns to group the data by. Based on the resulting groups, the aggregations are calculated.
        Possible groupings are e.g. by participant, recording date, or trial.
        To generate daily aggregations (the default), the ``groupby`` should contain the columns "subject_code"
        and "visit_date".
    unique_wb_id_column
        The name of the column (or index level) containing a unique identifier for every walking bout.
        The id does not have to be unique globally, but only within the groups defined by ``groupby``.
        Aka ``wb_dmos.reset_index().set_index([*groupby, unique_wb_id_column]).index.is_unique`` must be ``True``.

    Other Parameters
    ----------------
    %(other_parameters)s
    wb_dmos_mask
        A boolean DataFrame with the same shape the ``wb_dmos`` indicating the validity of every measure.
        Like the data, the ``wb_dmos_mask`` must have the ``groupby`` and the ``unique_wb_id_column`` as either as index
        or column available.
        After setting all of them as index, the index must be identical to the data.
        Every column of the data mask corresponds to a column of ``wb_dmos`` and has the same name.
        If an entry is ``False``, the corresponding measure is implausible and should be ignored for the aggregations.

        To exclude implausible data points from the input data, a ``wb_dmos_mask`` can be passed to the ``aggregate``
        method.
        The columns in ``wb_dmos_mask`` are regarded if there exists a column in the input data with the same name.
        Note that depending on which DMO measure is flagged as implausible, different elimination steps are applied:

        - "duration_s": The whole walking bout is not considered for the aggregation.
        - "n_steps": The corresponding "n_steps" is not regarded.
        - "n_turns": The corresponding "n_turns" is not regarded.
        - "walking_speed_mps": The corresponding "walking_speed_mps" is not regarded.
        - "stride_length_m": The corresponding "stride_length_m" AND the corresponding "walking_speed_mps" are not
          regarded.
        - "cadence_spm": The corresponding "cadence_spm" AND the corresponding "walking_speed_mps" are not regarded.
        - "stride_duration_s": The corresponding "stride_duration_s" is not regarded.

    Attributes
    ----------
    %(aggregated_data_)s
    filtered_wb_dmos_
        An updated version of ``wb_dmos`` with the implausible entries removed based on ``wb_dmos_mask``.
        ``filtered_wb_dmos_`` will have the groupby columns and the ``unique_wb_id_column`` set as index.

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
        "wb_all_sum": ("duration_s", "count"),
        "walkdur_all_sum": ("duration_s", "sum"),
        "steps_all_sum": ("n_steps", "sum"),
        "turns_all_sum": ("n_turns", "sum"),
        "wbdur_all_avg": ("duration_s", "median"),
        "wbdur_all_max": ("duration_s", _custom_quantile),
        "wbdur_all_var": ("duration_s", _coefficient_of_variation),
        "cadence_all_avg": ("cadence_spm", "mean"),
        "strdur_all_avg": ("stride_duration_s", "mean"),
        "cadence_all_var": ("cadence_spm", _coefficient_of_variation),
        "strdur_all_var": ("stride_duration_s", _coefficient_of_variation),
    }

    _TEN_THIRTY_WB_AGGS: typing.ClassVar = {
        "wb_1030_sum": ("duration_s", "count"),
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

    # TODO: Make that configurable somehow, as we should unify the units across the whole package
    _UNIT_CONVERSIONS: typing.ClassVar = {
        "walkdur_all_sum": 1 / 3600,
        "strlen_1030_avg": 100,
        "strlen_30_avg": 100,
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

    groupby: typing.Sequence[str]
    unique_wb_id_column: str

    wb_dmos_mask: pd.DataFrame

    filtered_wb_dmos_: pd.DataFrame

    def __init__(
        self,
        groupby: typing.Sequence[str] = ("visit_type", "participant_id", "measurement_date"),
        *,
        unique_wb_id_column: str = "wb_id",
    ) -> None:
        self.groupby = groupby
        self.unique_wb_id_column = unique_wb_id_column

    @base_aggregator_docfiller
    def aggregate(
        self,
        wb_dmos: pd.DataFrame,
        *,
        wb_dmos_mask: typing.Union[pd.DataFrame, None] = None,
        **_: Unpack[dict[str, typing.Any]],
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
        groupby = list(self.groupby)

        if not any(col in self.wb_dmos.columns for col in self.INPUT_COLUMNS):
            raise ValueError(f"None of the valid input columns {self.INPUT_COLUMNS} found in the passed dataframe.")

        if not all(col in self.wb_dmos.reset_index().columns for col in groupby):
            raise ValueError(f"Not all groupby columns {self.groupby} found in the passed dataframe.")

        data_correct_index = wb_dmos.reset_index().set_index([*groupby, self.unique_wb_id_column]).sort_index()

        if not data_correct_index.index.is_unique:
            raise ValueError(
                f"The passed data contains multiple entries for the same groupby columns {groupby}. "
                "Make sure that the passed data in `unique_wb_id_column` is unique for every groupby column "
                "combination."
            )

        if wb_dmos_mask is not None:
            wb_dmos_mask = (
                wb_dmos_mask.fillna(True)
                .infer_objects(copy=False)
                .reset_index()
                .set_index([*groupby, self.unique_wb_id_column])
                .sort_index()
            )

            if not wb_dmos.index.equals(wb_dmos_mask.index):
                raise ValueError(
                    "The data mask seems to be missing some data indices. "
                    "`wb_dmos_mask` must have exactly the same indices as `wb_dmos` after grouping."
                )

            wb_dmos_mask = wb_dmos_mask.reindex(data_correct_index.index)

            # We remove all individual elements from the data that are flagged as implausible in the data mask.
            self.filtered_wb_dmos_ = data_correct_index.where(wb_dmos_mask)
            # And then we need to consider some special cases:
            if "duration_s" in data_correct_index.columns and "duration_s" in wb_dmos_mask.columns:
                # If the duration is implausible, we need to remove the whole walking bout
                self.filtered_wb_dmos_ = self.filtered_wb_dmos_.where(wb_dmos_mask["duration_s"])
            if "walking_speed_mps" in data_correct_index.columns:
                walking_speed_filter = pd.Series(True, index=data_correct_index.index)
                # Walking speed is also implausible, if stride length or cadence are implausible
                if "stride_length_m" in wb_dmos_mask.columns:
                    walking_speed_filter &= wb_dmos_mask["stride_length_m"]
                if "cadence_spm" in wb_dmos_mask.columns:
                    walking_speed_filter &= wb_dmos_mask["cadence_spm"]
                self.filtered_wb_dmos_.loc[:, "walking_speed_mps"] = self.filtered_wb_dmos_.loc[
                    :, "walking_speed_mps"
                ].where(walking_speed_filter)
        else:
            self.filtered_wb_dmos_ = data_correct_index.copy()

        available_filters_and_aggs = self._select_aggregations(data_correct_index.columns)
        self.aggregated_data_ = self._apply_aggregations(self.filtered_wb_dmos_, groupby, available_filters_and_aggs)
        self.aggregated_data_ = self._fillna_count_columns(self.aggregated_data_)
        self.aggregated_data_ = self._convert_units(self.aggregated_data_)
        self.aggregated_data_ = self.aggregated_data_.round(3)

        return self

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
                    "because the data does not contain a 'duration_s' column.",
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
        groupby: list[str],
        available_filters_and_aggs: list[tuple[str, dict[str, tuple[str, typing.Union[str, typing.Callable]]]]],
    ) -> pd.DataFrame:
        """Apply filters and aggregations to the data."""
        aggregated_results = []
        for f, agg in available_filters_and_aggs:
            aggregated_data = filtered_data if f is None else filtered_data.query(f)
            aggregated_results.append(aggregated_data.groupby(groupby).agg(**agg))
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
