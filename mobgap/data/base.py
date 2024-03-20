"""Base classes for all fundamental dataset types."""

from typing import Any, NamedTuple

import pandas as pd
from tpcp import Dataset

from mobgap._docutils import make_filldoc

IMU_DATA_DTYPE = dict[str, pd.DataFrame]

base_gait_dataset_docfiller_dict = {
    "general_dataset_args": """
    groupby_cols
        Columns to group the data by. See :class:`~tpcp.Dataset` for details.
    subset_index
        The selected subset of the index. See :class:`~tpcp.Dataset` for details.
    """,
    "dataset_memory_args": """
    memory
        A joblib memory object to cache the results of the data loading.
        This is highly recommended, if you have many large data files.
        Otherwise, the initial index creation can take a long time.
    """,
    "common_dataset_data_attrs": """
    data
        The raw IMU data.
    sampling_rate_hz
        The sampling rate of the IMU data in Hz.
    participant_metadata
        The participant metadata loaded from the `infoForAlgo.mat` file.
    """,
    "common_dataset_reference_attrs": """
    reference_parameters_
        Parsed reference parameters.
        This contains the reference parameters in a format that can be used as input and output to many of the mobgap
        algorithms.
        See :func:`~mobgap.data.base.ReferenceData` for details.
        Note that these reference parameters are expected to be relative to the start of the recording and all timing
        parameters (like the start and end of a walking bout) are expected to be in samples.
    reference_parameters_relative_to_wb_
        Same as ``reference_parameters_``, but all timing parameters are relative to the start of the walking bout.
        This is useful for algorithms that only act in the context of a walking bout.
    reference_sampling_rate_hz_
        The sampling rate of the reference data in Hz.
    """,
    "dataset_classvars": """
    UNITS
        (ClassVar) Units of the IMU data
    """,
    "dataset_see_also": """
    :class:`~tpcp.Dataset`
        For details about the ``groupby_cols`` and ``subset_index`` parameters.
    """,
}
base_gait_dataset_docfiller = make_filldoc(
    base_gait_dataset_docfiller_dict,
    doc_summary="Decorator to fill common parts of the docstring for subclasses of :class:`BaseGaitDataset`.",
)


@base_gait_dataset_docfiller
class ReferenceData(NamedTuple):
    """Parsed reference parameters.

    All start/end values are provided in samples since the start of the recording.

    Attributes
    ----------
    wb_list
        A dataframe with the start and end of each walking bout in samples.
    ic_list
        A dataframe with the initial contacts in samples and the corresponding left/right label.
    turn_parameters
        A dataframe with the start, end, angle and other parameters of each turn.
    stride_parameters
        A dataframe with the start, end, duration and other parameters of each stride.

    """

    wb_list: pd.DataFrame
    ic_list: pd.DataFrame
    turn_parameters: pd.DataFrame
    stride_parameters: pd.DataFrame


@base_gait_dataset_docfiller
class BaseGaitDataset(Dataset):
    """Basic subclass for all normal gait datasets.

    This can be used as minimal interface for pipelines consuming gait data.

    Parameters
    ----------
    %(general_dataset_args)s

    Attributes
    ----------
    %(common_dataset_data_attrs)s
    %(dataset_classvars)s

    See Also
    --------
    %(dataset_see_also)s

    """

    class UNITS:
        """Representation of units IMU units in gait datasets.

        Parameters
        ----------
        acc
            acceleration unit, default = ms^-2
        gyr
            gyroscope unit, default = deg/s
        mag
            magnetometer unit, default = uT

        """

        acc: str = "ms^-2"
        gyr: str = "deg/s"
        mag: str = "uT"

    sampling_rate_hz: float
    data: IMU_DATA_DTYPE
    # TODO: Make that more specific, once we know what metadata is needed for the pipelines
    participant_metadata: dict[str, Any]


@base_gait_dataset_docfiller
class BaseGaitDatasetWithReference(BaseGaitDataset):
    """Base class for all gait datasets that have reference parameters.

    Parameters
    ----------
    %(general_dataset_args)s

    Attributes
    ----------
    %(common_dataset_data_attrs)s
    %(common_dataset_reference_attrs)s
    %(dataset_classvars)s

    See Also
    --------
    %(dataset_see_also)s
    """

    reference_sampling_rate_hz_: float

    reference_parameters_: ReferenceData
    reference_parameters_relative_to_wb_: ReferenceData


__all__ = [
    "BaseGaitDataset",
    "BaseGaitDatasetWithReference",
    "ReferenceData",
    "IMU_DATA_DTYPE",
    "base_gait_dataset_docfiller",
    "base_gait_dataset_docfiller_dict",
]
