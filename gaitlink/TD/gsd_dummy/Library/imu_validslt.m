function [valid,corectIMU] = imu_validslt(imudata)

% Get sensor position from environmental variable
SENSOR_POSITION = getenv_string('SENSOR_POSITION', 'LowerBack');

%[tempdata]=lowercase_structfileds(imudata);
% new_imudata.LowerBack=imudata.(imufileds{strcmp(lowerimufield,'lowerback')});
[tempdata]=imudata;

valid = 0;
corectIMU=[];
if isfield(tempdata, SENSOR_POSITION)
    [imudata]=lowercase_structfileds(tempdata.(SENSOR_POSITION));
    if isfield(imudata, 'acc') && isfield(imudata, 'gyr') && isfield(imudata, 'fs')

        [racc c] = size(imudata.acc);
        [rgyr c] = size(imudata.gyr);
        if racc~=0 && rgyr~=0
            valid = 1;

            corectIMU.(SENSOR_POSITION) = imudata;
        end
    end
end
end