function [output_struct, output_struct_json] = process_data(data, field_names, standard_string, bout_string)

% empty result variables that will be filled with the outputs
output_struct = struct();
output_struct_json = struct();

SENSOR_UNIT = getenv_string('SENSOR', 'SU');
SENSOR_POSITION = getenv_string('SENSOR_POSITION', 'LowerBack');

% iterate over the pre-extracted fieldnames for each relevant trial /
% recording
for i =1:length(field_names)
    % only proceed if there is actually data and not e.g. datetime
    if contains(field_names{i},'Recording') || contains(field_names{i},'Trial')
        disp(strcat('    ', field_names{i}));
        
        % split the name in its single parts
        names_split = strsplit(field_names{i},'.');
        
        % extract the data
        data_tmp = getfield(data,names_split{1:end});
        
        % check if SU even exists and write no output in case it is not 
        % available
        if isfield(data_tmp, SENSOR_UNIT)
            su = data_tmp.(SENSOR_UNIT);
        else
            continue
        end
        
        % correct potential upper case field names
        [~, su] = imu_validslt(su);
       
        % Check to see if there is any data in the recording
        if imu_valid(su)
            % if gold standard data is present use this for GSD
            % dummy
            if standards_present(data_tmp)
                reference_data_trial = data_tmp.Standards;
                
                gsd_result = GSD_Dummy(reference_data_trial,su,standard_string,bout_string);
            % if no gold standard data is present, save an empty
            % output
            else
                gsd_result = [];
                warning("No gold standard data available. Writing empty output");
            end
            % save the result into the correct output structure
            output_struct = set_field_GSD(output_struct,field_names{i},gsd_result);
            if length(gsd_result) == 1
                gsd_result = {gsd_result};
            end
            output_struct_json = set_field_GSD(output_struct_json, field_names{i},gsd_result);
        end
    end 
end
end