function [status] = driver(indir, outdir)
if (~isdeployed)
     addpath(genpath('./Library/'))
end

% load the data
load(fullfile(indir,'data.mat'));

% get the device for the gold standard according to what is available in
% the data.mat file
standard_string = getenv_string('STANDARD', 'Stereophoto');
disp(strcat('Standard: ', standard_string));

% get the type of gait bout: MicroWB, ContinuousWalkingPeriod, or Pass
% depending on what is available in the data.mat file
bout_string = getenv_string('BOUT', 'MicroWB');
disp(strcat('Bout type: ', bout_string));

% determine if the data set is from Free-living or Laboratory
% get the TimeMeasures in the data
time_measure_list = fieldnames(data);

% look at the first time measure
time_measure_i = data.(time_measure_list{1});
    
% get a list of recordings
recording_list = fieldnames(time_measure_i);

% check if there are "Recordings" or "Tests"
if all(contains(recording_list, 'Recording'))
    disp('Free living data set')
    field_names = fieldnamesr(data,2);
elseif all(contains(recording_list, 'Test'))
    disp('Lab data set')
    field_names = fieldnamesr(data,3);
else
    error('Data set not recognized as free-living or lab data set')
end

[output_struct, output_struct_json] = process_data(data, field_names, standard_string, bout_string);
                
%% save results

% Use the specified SENSOR_POSITION_OUTPUT_NAME
SENSOR_POSITION = getenv_string('SENSOR_POSITION_OUTPUT_NAME', 'LowerBack');
SENSOR_POSITION_OUTPUT_NAME = getenv_string('SENSOR_POSITION_OUTPUT_NAME', SENSOR_POSITION);

% .json file
output_struct_json = struct('GSD_Output',output_struct_json);
json_string = jsonencode(output_struct_json);
base_filename = strcat('GSD_Dummy_', standard_string, '_', bout_string);

% save json with regular name
filename = strcat(base_filename, '.json');
fid = fopen(fullfile(outdir,filename),'wt');
fprintf(fid,json_string);
fclose(fid);

% save json with generic name
fid = fopen(fullfile(outdir,'GSD_Output.json'),'wt');
fprintf(fid,json_string);
fclose(fid);

GSD_Output = output_struct;
% .mat file
filename = strcat(base_filename, '.mat');
save(fullfile(outdir,filename),'GSD_Output');
save(fullfile(outdir,'GSD_Output.mat'),'GSD_Output');
status = 'ok';

% Metadata
add_metadata(outdir, 'metadata.txt', base_filename);