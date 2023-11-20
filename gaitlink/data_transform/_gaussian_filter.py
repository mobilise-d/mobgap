import numpy as np
from scipy.ndimage import gaussian_filter
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

    data : np.ndarray
        The input data (assumed to be a NumPy array).

    Methods
    -------
    transform(data)
        Perform the blurring action on the input data.
    """

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma
        self.transformed_data_ = np.array([])  # Initialize transformed_data_ to an empty array

    def transform(self, data: np.ndarray) -> "GaussianFilter":
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
        if data is None:
            raise ValueError("Parameter 'data' must be provided.")

        self.data = data.copy()  # Create a copy for consistency
        # Apply Gaussian filter
        self.transformed_data_ = gaussian_filter(data, sigma=self.sigma, output=None, mode='reflect', cval=0.0, truncate=4.0)

        return self
