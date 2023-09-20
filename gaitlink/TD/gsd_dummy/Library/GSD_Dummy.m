function GSD = GSD_Dummy(reference,su,standard_string,bout_string)
% bool variables to plot results
plot_results  = strcmp(getenv_string('PLOT','false'), 'true');

% Get sensor unit and position from environmental variable
SENSOR_POSITION = getenv_string('SENSOR_POSITION', 'LowerBack');
SENSOR_UNIT = getenv_string('SENSOR', 'SU');

% get the available gold standards
reference_method_list = fieldnames(reference);

% check if selected gold standard is available in the data
if any(strcmp(reference_method_list,standard_string))
    % extract the reference data
    reference_data = reference.(standard_string);
    algorithm_output_fs = reference_data.Fs;

    % get the available bout types
    bout_type_list = fieldnames(reference_data);

    % check if selected bout type is available
    if any(strcmp(bout_type_list,bout_string))
        reference_data = reference_data.(bout_string);
    else
        warning('The bout type you selected is not available. Returning empty list.')
        GSD = [];
        return
    end
else
    warning('The gold standard method you selected is not available. Returning empty list.')
    GSD = [];
    return
end

if plot_results
    data_fs = su.(SENSOR_POSITION).fs;
    su_data = su.(SENSOR_POSITION).acc(:,1);
    figure;
    hold on
    plot(su_data ,'r')
end

% create an empty struct for the output
GSD = struct();

WB_start = [];
WB_end = [];
algorithm_output_fs = 0;

% extract the walking bouts from the reference data
for ref_bout = 1:size(reference_data,2)
    if isfield(reference_data,'Start')
        WB_start(ref_bout,:) = reference_data(ref_bout).Start;
        WB_end(ref_bout,:) = reference_data(ref_bout).End;
    end
end

% convert the output into the expected Mobilise-D format so it can be used
% by the subsequent blocks
GSD = build_gsd_output(GSD, WB_start, WB_end, algorithm_output_fs);

% plot results if set to true
if plot_results
    % extract walking from the output of the algorithm
    gs_boolean = zeros(size(su_data(:,1)));
    if isfield(GSD,'Start')
        for plot_i = 1:length(GSD)
            if GSD(plot_i).Start == 0
                gs_boolean(round(GSD(plot_i).Start*data_fs.Acc+1):round(GSD(plot_i).End*data_fs.Acc)) = max(su_data);
            else
                gs_boolean(round(GSD(plot_i).Start*data_fs.Acc):round(GSD(plot_i).End*data_fs.Acc)) = max(su_data);
            end
            plot(gs_boolean,'g', 'LineWidth',2)
        end
    end
    legend(['IMU ('  SENSOR_UNIT ', ' SENSOR_POSITION ')'],'GS Dummy', 'Interpreter', 'none')

end

% In case none the GSD field names are empty return empty result
if isempty(fieldnames(GSD))
    GSD = [];
end

end
