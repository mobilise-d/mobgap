function [acc, gyr, data_fs] = extract_imu_data(SU)

SENSOR_POSITION = getenv_string('SENSOR_POSITION', 'LowerBack');
SENSOR_UNIT = getenv_string('SENSOR', 'SU');

if isempty(SU)
    error(['Specified SENSOR_POSITION "', SENSOR_POSITION, '" of SENSOR_UNIT "', SENSOR_UNIT, '" contains no data.'])
end

if strcmp(SENSOR_POSITION,'LowerBack')
    if contains(SENSOR_UNIT,'2')
        acc = SU.LowerBack2.acc;
        gyr = SU.LowerBack2.gyr;
        data_fs = SU.LowerBack2.fs.Acc;
    else
        if isfield(SU, 'LowerBack')
            % Use CamelCase
            acc = SU.LowerBack.acc;
            gyr = SU.LowerBack.gyr;
            data_fs = SU.LowerBack.fs.Acc;
        else
            % Other style
            acc = SU.Lowerback.acc;
            gyr = SU.Lowerback.gyr;
            data_fs = SU.Lowerback.fs.Acc;
        end
    end
elseif strcmp(SENSOR_POSITION,'LeftWrist')
    if isfield(SU, 'LeftWrist')
        acc = SU.LeftWrist.acc;
        gyr = SU.LeftWrist.gyr;
        data_fs = SU.LeftWrist.fs.Acc;
    end
elseif strcmp(SENSOR_POSITION,'RightWrist')
    if isfield(SU, 'RightWrist')
        acc = SU.RightWrist.acc;
        gyr = SU.RightWrist.gyr;
        data_fs = SU.RightWrist.fs.Acc;
    end
else
    error(['Specified SENSOR_POSITION "', SENSOR_POSITION, '" of SENSOR_UNIT "', SENSOR_UNIT, '" is not implemented.'])
end

end