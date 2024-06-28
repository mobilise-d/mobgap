import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import Algorithm
from typing_extensions import Self

from mobgap._gaitmap.utils.rotations import rotate_dataset_series


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
        return rotate_dataset_series(self.data, self.orientation_object_[:-1])

    def estimate(
        self,
        data: pd.DataFrame,
        *,
        sampling_rate_hz: float,
    ) -> Self:
        """Estimate the orientation of the sensor based on the input data."""
        raise NotImplementedError("Needs to be implemented by child class.")
