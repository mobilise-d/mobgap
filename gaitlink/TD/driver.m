function [status] = driver(indir, outdir)
warning off
rmpath(genpath('./Library/'))
warning on
addpath(genpath('./Library/'))

% load the data
load(fullfile(indir,'data.mat'));

% load the GSD output
load(fullfile(indir,'GSD_Output.mat'));

% select the sensor unit to use. All options can be found in
% ./Library/process_data.m
sensor_string = getenv_string('SENSOR', 'SU');
disp(strcat('Sensor: ', sensor_string));


% determine if the data set is from Free-living or Laboratory
% get the TimeMeasures in the data
time_measure_list = fieldnames(data);

% look at the first time measure
time_measure_i = data.(time_measure_list{1});
    
% get a list of recordings
recording_list = fieldnames(time_measure_i);

% check if there are "Recordings" or "Tests", get the fieldnames to iterate
% over accordingly
if all(contains(recording_list, 'Recording'))
    disp('Free living data set')
    field_names = fieldnamesr(data,2);
elseif all(contains(recording_list, 'Test'))
    disp('Lab data set')
    field_names = fieldnamesr(data,3);
else
    error('Data set not recognized as free-living or lab data set')
end

% call one common function for free-living and lab data
[output_struct, output_struct_json] = process_data(data, GSD_Output, field_names, sensor_string);
%% save results
base_filename = strcat('TD_ElGohary', '_', sensor_string);

% .json file
output_struct_json = struct('TD_Output',output_struct_json);
json_string = jsonencode(output_struct_json);

% save json with regular name
filename = strcat(base_filename, '.json');
fid = fopen(fullfile(outdir,filename),'wt');
fprintf(fid,json_string);
fclose(fid);

% save json with generic name
fid = fopen(fullfile(outdir,'TD_Output.json'),'wt');
fprintf(fid,json_string);
fclose(fid);

% .mat file
TD_Output = output_struct;
filename = strcat(base_filename, '.mat');

% regular name
save(fullfile(outdir,filename),'TD_Output');

% generic name
save(fullfile(outdir,'TD_Output.mat'),'TD_Output');
status = 'ok';

% Metadata
add_metadata(outdir, 'metadata.txt', base_filename);
end