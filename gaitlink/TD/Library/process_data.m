function [output_struct, output_struct_json] = process_data(data, GSD_Output, field_names, sensor_string)

% empty output variables
output_struct = struct();
output_struct_json = struct();

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
        
        % determine which sensor data should be processed
        switch sensor_string
            case 'SU'
                % check if SU even exists and write no output in case it
                % is not available
                if isfield(data_tmp, 'SU')
                    su = data_tmp.SU;
                else
                    continue
                end
                    
            case 'SU_INDIP'
                su = data_tmp.SU_INDIP;
            % INDIP and INDIP2 will only be differentiated in the
            % preprocessing of the respective algorithm, where the actual
            % acc and gyro data is extracted
            case 'SU_INDIP2'
                su = data_tmp.SU_INDIP;
            case 'SU_Axivity'
                su = data_tmp.SU_Axivity;
            otherwise
                error('Selected sensor_string %s unknown.',sensor_string)            
        end
        
        % correct potential upper case field names
        [~, su] = imu_validslt(su);
        
        % Check to see if there is any data in the recording
        if imu_valid(su)
            % extract the IMU gait sequences from GSD_Output.mat
            gs_trial = getfield(GSD_Output,names_split{1:end},'SU');
            % run the turning algorithm
            td_result = TD_ElGohary(su, gs_trial, sensor_string);
        end
        
        % save the output to the correct field in the output struct
        output_struct = set_field_TD(output_struct,field_names{i},td_result);
        
        % fix json output in case of length 1
        if length(td_result) == 1
            td_result = {td_result};
        end
        output_struct_json = set_field_TD(output_struct_json, field_names{i},td_result);
    end
end
end