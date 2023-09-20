function valid = imu_valid(SU)

[acc, gyr] = extract_imu_data(SU);

[r c] = size(acc);
valid = 1;
if r==0
    valid = 0;
end

[r c] = size(gyr);
if r==0
    valid = 0;
end