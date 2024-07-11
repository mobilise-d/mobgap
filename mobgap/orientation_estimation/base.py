"""Base classes for the orientation estimation methods that can be used to estimate the orientation of an IMU."""

from typing import Any

import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import Algorithm
from typing_extensions import Self, Unpack

from mobgap.utils.conversions import transform_to_global_frame


class BaseOrientationEstimation(Algorithm):
    """Base class for the individual Orientation estimation methods that work on pd.DataFrame data."""

    _action_methods = ("estimate",)
    orientation_object_: Rotation

    data: pd.DataFrame
    sampling_rate_hz: float

    @property
    def orientation_(self) -> pd.DataFrame:
        """Orientations as pd.DataFrame."""
        df = pd.DataFrame(self.orientation_object_.as_quat(), columns=["q_x", "q_y", "q_z", "q_w"])
        df.index.name = "sample"
        return df

    @property
    def rotated_data_(self) -> pd.DataFrame:
        """Rotated data."""
        return transform_to_global_frame(self.data, self.orientation_object_[:-1])

    def estimate(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Estimate the orientation of the sensor based on the input data."""
        raise NotImplementedError("Needs to be implemented by child class.")
