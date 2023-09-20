


function [tempdata]=lowercase_structfileds(imudata)

    imufileds=fieldnames(imudata);
    lowerimufield=lower(imufileds);
    for i=1:length(imufileds)
        tempdata.(lowerimufield{i})=imudata.(imufileds{i});
    end

end