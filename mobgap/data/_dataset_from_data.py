from collections.abc import Hashable
from copy import copy
from typing import Any, Optional, Union

import pandas as pd

from mobgap.data.base import IMU_DATA_DTYPE, BaseGaitDataset, base_gait_dataset_docfiller

KEY_DTYPE = Union[tuple[Hashable, ...], Hashable]


@base_gait_dataset_docfiller
class GaitDatasetFromData(BaseGaitDataset):
    """Create dataset from data that is already loaded.

    This is useful for testing and running pipelines on individual one-off datasets.

    Parameters
    ----------
    _data
        The IMU data, as a dictionary of dictionaries of dataframes.
        The first level of the dictionary is the group label (e.g. participant id), the second level is the sensor
        label (e.g. "LowerBack").
        The first level keys are turned into the index of the dataset.
        If you want to have multiple columns in the dataset index, use a tuple as the key.
        To customize the names of the index columns, use the ``index_cols`` parameter.
    _sampling_rate_hz
        The sampling rate of the IMU data in Hz.
        If you have different sampling rates for different groups, you can pass a dictionary with the group label as
        key and the sampling rate as value.
    _participant_metadata
        Metadata for each group.
        The keys of the dictionary are expected to be the same as the keys of the ``_data`` dictionary.
        The content of the metadata is not specified and can be anything.
    index_cols
        The name of the index columns.
        If your data keys are tuples, you can pass a list of strings to name the index columns.
    %(general_dataset_args)s

    Attributes
    ----------
    %(common_dataset_data_attrs)s

    Notes
    -----
    To avoid creating copies of your data, this dataset will shallow copy the data and participant metadata when
    ``~tpcp.clone`` is called on a dataset instance.
    This happens a couple of times in ``tpcp`` internal code, so we felt the need to reduce the memory footprint of
    this dataset.
    Just keep in mind that this means that if you modify the data or metadata of a dataset, you will also modify the
    original data and metadata.
    But, you should not modify the data or metadata of a dataset anyway, so that should be fine.

    See Also
    --------
    %(dataset_see_also)s

    """

    def __init__(
        self,
        _data: dict[KEY_DTYPE, dict[str, pd.DataFrame]],
        _sampling_rate_hz: Union[float, dict[KEY_DTYPE, float]],
        _participant_metadata: Optional[dict[KEY_DTYPE, dict[str, Any]]] = None,
        *,
        index_cols: Optional[Union[list[str], str]] = None,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self._data = _data
        self._sampling_rate_hz = _sampling_rate_hz
        self._participant_metadata = _participant_metadata
        self.index_cols = index_cols

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    @property
    def data(self) -> IMU_DATA_DTYPE:
        self.assert_is_single(None, "data")

        return self._data[self._group_label_correct_format]

    @property
    def sampling_rate_hz(self) -> float:
        self.assert_is_single(None, "sampling_rate_hz")

        sampling_rate = self._sampling_rate_hz

        if isinstance(sampling_rate, (int, float)):
            return sampling_rate

        return sampling_rate[self._group_label_correct_format]

    @property
    def participant_metadata(self) -> dict[str, Any]:
        self.assert_is_single(None, "participant_metadata")

        if self._participant_metadata is None:
            raise AttributeError("No participant metadata provided for this dataset")

        return self._participant_metadata[self._group_label_correct_format]

    @property
    def _is_tuple_key(self) -> bool:
        return isinstance(next(iter(self._data.keys())), tuple)

    @property
    def _group_label_correct_format(self) -> Union[Hashable, tuple[Hashable, ...]]:
        return self.group_label if self._is_tuple_key else self.group_label[0]

    def create_index(self) -> pd.DataFrame:
        index_cols = [self.index_cols] if isinstance(self.index_cols, str) else self.index_cols
        return pd.DataFrame(list(self._data.keys()), columns=index_cols)

    @classmethod
    def __clone_param__(cls, param_name: str, value: Any) -> Any:
        # To avoid copying the data, we shallow clone the data and participant_metadata
        if param_name in ["_data", "_participant_metadata"]:
            return copy(value)
        return super().__clone_param__(param_name, value)
