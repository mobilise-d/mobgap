


function [valid,corectIMU] = imu_validslt(imudata)

    [tempdata]=lowercase_structfileds(imudata);

    % new_imudata.LowerBack=imudata.(imufileds{strcmp(lowerimufield,'lowerback')});

    valid = 0;
    corectIMU=[];
    if isfield(tempdata, 'lowerback')
        [lowerbackdata]=lowercase_structfileds(tempdata.lowerback);
        if isfield(lowerbackdata, 'acc') && isfield(lowerbackdata, 'gyr') && isfield(lowerbackdata, 'fs')

            [racc c] = size(lowerbackdata.acc);
            [rgyr c] = size(lowerbackdata.gyr);
            if racc~=0 && rgyr~=0
                valid = 1;
                
                corectIMU.LowerBack = lowerbackdata;
                                
            end

        end

    end

end