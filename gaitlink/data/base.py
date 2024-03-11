from typing import Any, NamedTuple

import pandas as pd
from tpcp import Dataset

IMU_DATA_DTYPE = dict[str, pd.DataFrame]


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


class BaseGaitlinkDataset(Dataset):

    sampling_rate_hz: float
    data: IMU_DATA_DTYPE
    # TODO: Make that more specific, once we kniw what metadata is needed for the pipelines
    participant_metadata: dict[str, Any]


class BaseGaitlinkDatasetWithReference(BaseGaitlinkDataset):
    reference_sampling_rate_hz_: float

    reference_parameters_: ReferenceData
    reference_parameters_relative_to_wb_: ReferenceData
