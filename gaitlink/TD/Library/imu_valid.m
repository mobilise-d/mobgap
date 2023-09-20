function valid = imu_valid(IMU)

if isfield(IMU, 'LowerBack')
   % Use CamelCase
    acc = IMU.LowerBack.acc;
    gyr = IMU.LowerBack.gyr;
    data_fs = IMU.LowerBack.fs;
else
    % Other style
    acc = IMU.Lowerback.acc;
    gyr = IMU.Lowerback.gyr;
    data_fs = IMU.Lowerback.fs;
end

[r c] = size(acc);
valid = 1;
if r==0
    valid = 0;
end

[r c] = size(gyr);
if r==0
    valid = 0;
end