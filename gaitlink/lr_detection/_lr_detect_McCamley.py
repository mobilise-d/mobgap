import numpy as np
import pandas as pd
from typing import Optional

from gaitlink.data_transform import ButterworthFilter
from gaitlink.lr_detection.base import BaseLRDetector, base_lr_detector_docfiller

# TODO: Update docstrings.

class McCamleyLRDetection(BaseLRDetector):
    """
    This class uses the McCamley algorithm (later improved in Ullrich et al.) in order to predict whether each pre-determined initial contact (IC) corresponds to a left or a right step.

    In the original McCamley algorithm, the angular velocity around the vertical axis ("gyr_x") serves as the distinguishing factor for identifying left and right ICs. The process involves the following steps:

        * Signal Pre-processing: Subtracting the signal mean and applying a low-pass filter (4th order Butterworth filter with a 2 Hz cut-off frequency).
        * IC Assignment: Analyzing the sign of the filtered "gyr_x" value at the IC time point for classification. If the value is positive, the IC is attributed to the right foot; if negative, it's attributed to the left foot.

    As a first extension to the original McCamley algorithm, the angular velocity around the anterior-posterior axis, "gyr_z", can resemble a periodic wave with a constant phase shift w.r.t. "gyr_x" after application of the low-pass filter described above. This is also a suitable input signal for the McCamley algorithm, when inverting the sign. A second and final extension to the original McCamley algorithm is to use the combination of the filtered signals for the vertical and anterior-posterior signals: gyr_comb = gyr_x - gyr_z.
:
    The methodology used here is based on the following reference papers:
    1) J. McCamley et al., “An enhanced estimate of initial contact and final contact instants of time using lower trunk inertial sensor data,” Gait & posture, 2012, available at: 
    https://www.sciencedirect.com/science/article/pii/S0966636212000707?via%3Dihub

    2) Reference Papers: Ullrich M, Kuderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021, available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653
    """

    def __init__(self,
                axis_config: str = "Combined",
                butter_filter: Optional[ButterworthFilter] = None):
        """
        Initializes the McCamleyLRDetection instance.

        Attributes:
        ---------------
            axis_config (str): The axis configuration. Default is "Combined". Other options are "Vertical" and "Anterior-Posterior".
            butter_filter (Optional[ButterworthFilter]): The Butterworth filter to use. If None, a default filter is used.
        """
        self.axis_config  = axis_config
        self.butter_filter = butter_filter

        # This is done to order to avoid a mutable default argument, which can lead to unexpected behaviour.
        if self.butter_filter is None:
            self.butter_filter = ButterworthFilter(order = 4, cutoff_freq_hz=(0.5, 2), filter_type = 'bandpass')

    @base_lr_detector_docfiller
    def detect(self,
               data: pd.DataFrame,
               ic_list: pd.DataFrame,
               sampling_rate_hz: float = 100,
               ):
        
        """
        %(detect_)s

        """
        self.sampling_rate_hz = sampling_rate_hz

        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")
            
        if not isinstance(ic_list, pd.DataFrame):
            raise TypeError("'ic_list' must be a pandas DataFrame")
        
        # create a copy of ic_list, otherwise they will get modified when edge cases are detected.
        ic_list = ic_list.copy()
        
        # check axis configuration
        if self.axis_config.upper() not in ['VERTICAL', 'V', 'YAW', 'Y', 'ANTERIOR-POSTERIOR', 'AP', 'ROLL', 'R', 'COMBINED', 'C']:
            raise NotImplementedError("The axis configuration you have selected is not supported. Please select between 'Vertical', 'Anterior-Posterior' or 'Combined'.")


        if self.axis_config.upper() in ['VERTICAL', 'V', 'YAW', 'Y']:
            data_per_gs = data["gyr_x"].copy()
        elif self.axis_config.upper() in ['ANTERIOR-POSTERIOR', 'AP', 'ROLL', 'R']:
            data_per_gs = data["gyr_z"].copy() * -1 # invert the sign here
        elif self.axis_config.upper() in ['COMBINED', 'C']:
            data_per_gs = data["gyr_z"].copy() * - 1 + data["gyr_x"].copy()

        # due to rounding errors the last two ICs might need to be shifted backwards
        if ic_list["ic"].iloc[-1] >= len(data_per_gs):
            # print("edge case 1")
            ic_list.loc[ic_list.index[-1], "ic"] = len(data_per_gs) - 1
        if len(ic_list) >= 2 and ic_list["ic"].iloc[-2] >= len(data_per_gs):
            # print("edge case 2")
            ic_list.loc[ic_list.index[-2], "ic"] = len(data_per_gs) - 2

        # We could have also implemented the original McCamley algorithm here. However, this is deprecated. The original algo subtracted the mean from the signal and used a low_pass filter. This was not used in the TVS. The version below is using a bandpass filter instead of subtracting the mean to make it more robust for turnings.

        data_filtered = self.butter_filter.filter(data_per_gs, sampling_rate_hz = self.sampling_rate_hz).filtered_data_

        prediction_per_gs = pd.Series(np.where(data_filtered.iloc[ic_list["ic"].values] <= 0, "Left", "Right")).reset_index(drop=True)

        # keep track of the predicted labels and the processed data
        self.ic_lr =  prediction_per_gs.rename("predicted_lr_label").to_frame()
        self.processed_data = data_filtered
                

        return self