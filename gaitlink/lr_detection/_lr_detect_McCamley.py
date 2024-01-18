import numpy as np
import pandas as pd

from gaitlink.data_transform import ButterworthFilter


from gaitlink.lr_detection.base import BaseLRDetector, base_lr_detector_docfiller

class McCamleyLRDetection(BaseLRDetector):
    """
    Left/Right foot detector based on McCamley approach [insert reference here].
    """

    def __init__(self,
                 axis_config: str = "Combined",
                 sampling_rate_hz: float = 100,
                 lower_band: float = 0.5,
                 upper_band: float = 2,
                 order: int = 4):
        """
        Parameters:
        ----------------
        sampling_rate_hz (float):  
        axis_config (str):
        lower_band (float):
        upper_band (float): 
        order (int):
        """
        
        # filtering parameters
        self.lower_band = lower_band
        self.upper_band = upper_band
        self.order = order
        self.axis_config  = axis_config
        self.sampling_rate_hz = sampling_rate_hz



    @base_lr_detector_docfiller
    def detect(self,
               data: pd.DataFrame,
               ics: pd.DataFrame,
               ):
        
        """
        Parameters:
        ----------------
        data (pd.DataFrame): A dataframe representing data from a GS.
        ics (pd.Series): A series representing a list of ICs within a GS, 0-index at the start of the GS.

        Returns:
        ----------------
        ic_left_right
        """
        # # check for types of data, do not allow convenience conversions.
        # if not isinstance(data, pd.DataFrame):
        #     raise TypeError("data must be a pandas DataFrame")
            
        # if not isinstance(ics, pd.Series):
        #     raise TypeError("ics must be a pandas Series")

        
        # check axis configuration
        if self.axis_config.upper() not in ['VERTICAL', 'V', 'YAW', 'Y', 'ANTERIOR-POSTERIOR', 'AP', 'ROLL', 'R', 'COMBINED', 'C']:
            raise NotImplementedError("The axis configuration you have selected is not supported. Please select between 'Vertical', 'Anterior-Posterior' or 'Combined'.")

        if self.axis_config.upper() in ['VERTICAL', 'V', 'YAW', 'Y']:
            data = data["gyr_x"].copy()
        elif self.axis_config.upper() in ['ANTERIOR-POSTERIOR', 'AP', 'ROLL', 'R']:
            data = data["gyr_z"].copy() * -1 # invert the sign here
        elif self.axis_config.upper() in ['COMBINED', 'C']:
            data = data["gyr_z"].copy() * - 1 + data["gyr_x"].copy()

        
        if ics["ic"].iloc[-1] >= len(data):
            print("edge case 1")
            ics["ic"].iloc[-1] = len(data) - 1
        if len(ics) >= 2 and ics["ic"].iloc[-2] >= len(data):
            print("edge case 2")
            ics["ic"].iloc[-2] = len(data) - 2

        # TODO: we can also implement the original McCamley algorithm here. However, this is deprecated. The original algo subtracted the mean from the signal and used a low_pass filter. This was not used in the TVS. The version below is using a bandpass filter instead of subtracting the mean to make it more robust for turnings.

        butter_filter = ButterworthFilter(order = self.order, cutoff_freq_hz = (self.lower_band, self.upper_band), filter_type="bandpass") 
        data_filtered = butter_filter.filter(data, sampling_rate_hz = self.sampling_rate_hz).filtered_data_

        self.ic_lr = data_filtered.iloc[ics["ic"].values].apply(lambda x: "Left" if x <= 0 else "Right").reset_index(drop=True)
        self.ic_lr =  self.ic_lr.rename("predicted_lr_label").to_frame()

        return self