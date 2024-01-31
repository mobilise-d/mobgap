import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Self
from gaitlink.utils.dtypes import DfLike, dflike_as_2d_array
from gaitlink.data_transform.base import BaseTransformer


class GaussianFilter(BaseTransformer):
    """
    A class for applying Gaussian filter to blur images or multidimensional data.

    Derived from BaseTransformer for integration into gait analysis pipelines.


    Parameters
    ----------
    sigma : float
        Standard deviation for Gaussian kernel.

    Attributes
    ----------
    transformed_data_ :
        The data after applying the Gaussian filter.

    Other Parameters
    ----------------
    data
        The input data

    """

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def transform(self, data: np.ndarray) -> Self:
        """
        Apply the Gaussian filter to blur the input data.

        Parameters
        ----------
        data : np.ndarray
            A NumPy array representing image or multidimensional data.

        Returns
        -------
        GaussianFilter
            The instance of the transform with the blurred results attached.

        """

        self.data = data

        # Convert to 2D array using dflike_as_2d_array function
        df_data, index, transformation_function = dflike_as_2d_array(data)

        # Apply Gaussian filter
        self.transformed_data_ = gaussian_filter1d(
            df_data, sigma=self.sigma, axis=0, output=None, mode="reflect", cval=0.0, truncate=4.0
        )

        # Back transformation using the transformation function
        self.transformed_data_ = transformation_function(self.transformed_data_, index)

        return self
