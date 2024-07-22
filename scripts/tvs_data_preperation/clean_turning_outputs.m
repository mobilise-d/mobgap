% Define data folder path (modify with your actual path)
dataFolder = '/home/arne/Documents/repos/private/mobilised_tvs_data/tvs_dataset';

% Fields to remove from INDIP and Stereo standard (modify if needed)
fieldsToDelete = {'Turn_Start', 'Turn_End', 'Turn_Duration', ...
    'Turning_Flag', 'Turn_Number', 'Turn_Angle', ...
    'Turn_NumberStrides', 'Turn_AngularVelocity', ...
    'Turn_PeakAngularVelocity', 'Turn_MeanAngularVelocity', ...
    'Turn_Length'};

% Get all cohort subfolders
cohortFolders = dir(dataFolder); % Modify with your actual cohort folder pattern
cohortFolders = cohortFolders([cohortFolders.isdir] & ~ismember({cohortFolders.name}, {'.', '..'}));

% Loop through each cohort folder
for i = 1:length(cohortFolders)
    cohortFolder = fullfile(dataFolder, cohortFolders(i).name);   
    cohortName = cohortFolders(i).name; % Get cohort folder name
    % Get all subject subfolders within the cohort
    subjectFolders = dir(cohortFolder);
    subjectFolders = subjectFolders([subjectFolders.isdir] & ~ismember({subjectFolders.name}, {'.', '..'}));

    % Loop through each subject folder
    for j = 1:length(subjectFolders)
        subjectFolder = fullfile(cohortFolder, subjectFolders(j).name);
        subjectID = subjectFolders(j).name; % Get subject folder name
        % Free-living data processing
        freelivingFile = fullfile(subjectFolder, 'Free-living', 'data.mat');
        freelivingFile2 = fullfile(subjectFolder, 'Free-living');
        % Create "No turns" folder if it doesn't exist
        if ~exist(freelivingFile2, 'dir')
            mkdir(freelivingFile2)
        end
        % Process free-living data if the file exists
        if exist(freelivingFile, 'file')
            fprintf('Processing: Cohort - %s, Subject - %s, Acquisition - Free-living\n', cohortName, subjectID);
            data_head = load(freelivingFile);
            
            data = data_head.data;
            
            if isfield(data.TimeMeasure1, "Recording4") && ...
               isfield(data.TimeMeasure1.Recording4, 'Standards')

                standards = data.TimeMeasure1.Recording4.Standards;

                % Check if the INDIP standard data exists
                if isfield(standards, 'INDIP')
                    % Access INDIP standard data
                    INDIP_standard = standards.INDIP;

                    % in case of empty fields
                    if ~isempty(INDIP_standard.MicroWB)
                        INDIP_standard.MicroWB = delete_fields(INDIP_standard.MicroWB, fieldsToDelete);
                    else
                        fprintf('\t- MicroWB is empty\n');
                    end
                    if ~isempty(INDIP_standard.ContinuousWalkingPeriod)
                        INDIP_standard.ContinuousWalkingPeriod = delete_fields(INDIP_standard.ContinuousWalkingPeriod, fieldsToDelete);
                    else
                        fprintf('\t- ContinuousWalkingPeriod is empty\n');
                    end
                    data.TimeMeasure1.Recording4.Standards.INDIP = INDIP_standard;
                else
                    % Handle the case where INDIP_standard does not exist
                    fprintf('\t- %s, %s, No INDIP Reference\n', test_names{ii}, trial_names{jj});
                end
            else
                fprintf('\t- %s, %s, No Reference Information\n', test_names{ii}, trial_names{jj});
            end

            save(fullfile(freelivingFile2,'data.mat'), "data");
        end

        % Laboratory data processing
        labFile = fullfile(subjectFolder, 'Laboratory', 'data.mat');
        labFile2 = fullfile(subjectFolder, 'Laboratory');
        % Create "No turns" folder if it doesn't exist
        if ~exist(labFile2, 'dir')
            mkdir(labFile2)
        end
        % Process free-living data if the file exists
        if exist(labFile, 'file')
            fprintf('Processing: Cohort - %s, Subject - %s, Acquisition - Laboratory\n', cohortName, subjectID);
            data_head = load(labFile);
            
            data = data_head.data;

            % Iterate through tests (Test 1, Test 2 and Test 3 excluded)
            test_names = fieldnames(data.TimeMeasure1);
            for ii = 4:length(test_names)
                % Iterate through all trials
                trial_names = fieldnames(data.TimeMeasure1.(test_names{ii}));
                for jj = 1:length(trial_names)
                    % Access INDIP standard data
                    if isfield(data.TimeMeasure1, test_names{ii}) && ...
                       isfield(data.TimeMeasure1.(test_names{ii}), trial_names{jj}) && ...
                       isfield(data.TimeMeasure1.(test_names{ii}).(trial_names{jj}), 'Standards')

                        standards = data.TimeMeasure1.(test_names{ii}).(trial_names{jj}).Standards;

                        % Check if the INDIP standard data exists
                        if isfield(standards, 'INDIP')
                            % Access INDIP standard data
                            INDIP_standard = standards.INDIP;
                            if ~isempty(INDIP_standard.MicroWB)
                                INDIP_standard.MicroWB = delete_fields(INDIP_standard.MicroWB, fieldsToDelete);
                            else
                                fprintf('\t- %s, %s, INDIP- MicroWB is empty\n', test_names{ii}, trial_names{jj});
                            end
                            if ~isempty(INDIP_standard.ContinuousWalkingPeriod)
                                INDIP_standard.ContinuousWalkingPeriod = delete_fields(INDIP_standard.ContinuousWalkingPeriod, fieldsToDelete);
                            else
                                fprintf('\t- %s, %s, INDIP - ContinuousWalkingPeriod is empty\n', test_names{ii}, trial_names{jj});
                            end
                            data.TimeMeasure1.(test_names{ii}).(trial_names{jj}).Standards.INDIP = INDIP_standard;
                        else
                            % Handle the case where INDIP_standard does not exist
                            fprintf('\t- %s, %s, No INDIP Reference\n', test_names{ii}, trial_names{jj});
                        end

                        % Check if the Stereo standard data exists
                        if isfield(standards, 'Stereophoto')
                            % Access Stereo standard data
                            Stereo_standard = standards.Stereophoto;
                              
                            if ~isempty(Stereo_standard.MicroWB)
                                Stereo_standard.MicroWB = delete_fields(Stereo_standard.MicroWB, fieldsToDelete);
                            else
                                fprintf('\t- %s, %s, Stereo - MicroWB is empty\n', test_names{ii}, trial_names{jj});
                            end
                            if ~isempty(Stereo_standard.ContinuousWalkingPeriod)
                                Stereo_standard.ContinuousWalkingPeriod = delete_fields(Stereo_standard.ContinuousWalkingPeriod, fieldsToDelete);
                            else
                                fprintf('\t- %s, %s, Stereo - ContinuousWalkingPeriod is empty\n', test_names{ii}, trial_names{jj});
                            end
                            data.TimeMeasure1.(test_names{ii}).(trial_names{jj}).Standards.Stereophoto = Stereo_standard;
                        else
                           fprintf('\t- %s, %s, No Stereophoto\n', test_names{ii}, trial_names{jj});
                        end
                    else
                        fprintf('\t- %s, %s, No Reference Information\n', test_names{ii}, trial_names{jj});
                    end

                end
            end
            % Remove specified fields and save data
            save(fullfile(labFile2,'data.mat'), "data");
        end
    end
end


function s = delete_fields(s, fields)
    % s: the input struct
    % fields: a cell array of strings representing the field names to delete

    for i = 1:length(fields)
        if isfield(s, fields{i})
            s = rmfield(s, fields{i});
        end
    end
end
