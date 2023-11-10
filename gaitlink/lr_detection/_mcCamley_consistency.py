import numpy as np
from gaitlink.lr_detection._utils import _butter_lowpass_filter, find_extrema_in_radius


def identify_left_right_ullrich(gyr_yaw,
                                ic_list,
                                sampling_rate,
                                version):
    """experimental function to make LR detection more robust, not finalized and not used in TVS"""

    ic_list_samples = np.round(ic_list * sampling_rate).astype(int)

    if ic_list_samples[-1] >= len(gyr_yaw):
        ic_list_samples[-1] = len(gyr_yaw)-1

    lowpass_cut_off_hz = 2
    gyr_yaw = _butter_lowpass_filter(gyr_yaw, lowpass_cut_off_hz, sampling_rate)

    # TODO: fine tune this parameter
    radius_left = 30
    radius_right = 15

    if version == "snap_to_extremum":
        # find extrema of absolute signal via function
        gyr_yaw_abs = abs(gyr_yaw)

        extrema = find_extrema_in_radius(gyr_yaw_abs, ic_list_samples, radius_left, radius_right, "max")
        ic_left_right_improved = np.where(gyr_yaw[extrema]<=0, "L", "R")

        return ic_left_right_improved, extrema, radius_left, radius_right

    elif version == "consistency_check":
        # This is currently to working.

        ic_left_right = np.where(gyr_yaw[ic_list_samples]<=0, "L", "R")
        ic_left_right_01 = np.where(gyr_yaw[ic_list_samples]<=0, 0, 1)

        # TODO: idea: check where the L R L R pattern is broken and only improve in those situations
        ic_left_right_01_diff = np.diff(ic_left_right_01)
        # the diff is 0 if two subsequent steps are assigned to the same foot
        consistency_check = np.where(ic_left_right_01_diff == 0)[0]

        if consistency_check.size == 0:
            return ic_left_right

        k = 1
        max_loops = 100 # to ensure that the loop does not run forever
        while consistency_check.size != 0  and k < max_loops:

            print(np.where(gyr_yaw[ic_list_samples]<=0, 0, 1))
            print(np.diff(np.where(gyr_yaw[ic_list_samples]<=0, 0, 1)))
            print("\n")

            # find extrema of absolute signal via function
            gyr_yaw_abs = abs(gyr_yaw)

            extrema = find_extrema_in_radius(gyr_yaw_abs, ic_list_samples, radius_left, radius_right, "max")
            ic_left_right_improved = np.where(gyr_yaw[extrema]<=0, "L", "R")

            ic_left_right_improved_01 = np.where(gyr_yaw[extrema]<=0, 0, 1)
            ic_left_right_improved_01_diff = np.diff(ic_left_right_improved_01)

            consistency_check = np.where(ic_left_right_improved_01_diff == 0)[0]
            
            k += 1
        
        return ic_left_right_improved