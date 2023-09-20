function imu_preprocessed = preprocessing_standardized(IMU, gs_start_sample, gs_end_sample,sensor_string)
    % check if LowerBack2 should be used
    if contains(sensor_string,'2')    
        imu_preprocessed = IMU.LowerBack2.gyr(gs_start_sample:gs_end_sample,1);
    else
        imu_preprocessed = IMU.LowerBack.gyr(gs_start_sample:gs_end_sample,1);
    end

