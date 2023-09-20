%% [0] Intialise
clear all;
close all;
clc;
%% [1]
% 1.1 Set directorys
% Make sure your current working directory is the directory of this file
currentwd = ('H:\Mobilise-D-TVS-Recommended-Algorithms\TD_ElGohary');
cd(currentwd)
% addpath(genpath('.'));

% Add folder with algorithm code to path
librarydir = ('H:\Mobilise-D-TVS-Recommended-Algorithms');
% addpath(genpath(librarydir));

% Set general pipeline parameters
% Data directory

datadir = 'H:\gaitlink\example_data\data\lab';

% Example for one participant.
% TODO: loop through all participant diretories
indir  = (datadir);

% [1.2] Get cohort info to start loop
cd(datadir);
Cohort_dir = dir;
Cohort_Folders = Cohort_dir(3:end);
Cohort_names = extractfield(Cohort_Folders,'name')';
Cohort_Size = size(Cohort_names,1);

for iCohort = 1:Cohort_Size
    CohortNowChar = cell2mat(Cohort_names(iCohort));
    DirectoryPathCohort = [datadir, '\',CohortNowChar];
    cd(DirectoryPathCohort);
    
    Subject_dir = dir;
    Subject_Folders = Subject_dir(3:end);
    Subject_names = extractfield(Subject_Folders,'name')';
    Subject_Size = size(Subject_names,1);
    
    
    % loop over all study participants
    for iSubject = 1:Subject_Size
        % Extract subject name
        SubjectNowChar = cell2mat(Subject_names(iSubject));
        SubjectNow = cell2mat(Subject_names(iSubject));
        SubjectCode = SubjectNowChar(4:end);
        fprintf(1, 'Now reading %s\n', SubjectNowChar);
        
        % Set directory to subject folder created by CiC_Standardization
        DirectoryPathSubjectGait = [DirectoryPathCohort, '\',SubjectNow];
        cd(DirectoryPathSubjectGait);

        % Get names of files to copy over
        SubjectData_Name = [DirectoryPathSubjectGait, '\', 'data.mat'];
        SubjectInfo_Name = [DirectoryPathSubjectGait, '\', 'infoForAlgo.mat'];

        outdir = DirectoryPathSubjectGait;
        indir  = (DirectoryPathSubjectGait);
        
        % [1.4] Setup pipeline
        % Define environment variables
        SENSOR_UNIT     = 'SU'; % SU, SU_INDIP, SU_INDIP2, SU_Axivity
        SENSOR_POSITION = 'LowerBack'; %'LowerBack';  %'Wrist'; DEFAULT: 'LowerBack'; handedness is read out later
        
        % Those variable are used to define the SU and Position names in the
        % results output files. E.g., set them to 'SU', 'LowerBack', if you want to
        % have the output struct fields always set to those values
        % Default: Use the same as SENSOR_UNIT and SENSOR_POSITION
        SENSOR_UNIT_OUTPUT_NAME = SENSOR_UNIT;
        SENSOR_POSITION_OUTPUT_NAME = SENSOR_POSITION;
        
        % Reference information
        STANDARD_UNIT = 'INDIP'; % Choose reference system: INDIP, Stereophoto, Walkway, IMU, Gaitrite,SU_LowerShanks
        STANDARD_BOUT = 'MicroWB'; % Choose reference bout type: MicroWB, ContinuousWalkingPeriod, Pass,
        
        % Plot intermediate results?
        BLOCK_PLOT = 'false';
        
        setenv('SENSOR', SENSOR_UNIT);
        setenv('SENSOR_POSITION',SENSOR_POSITION);
        setenv('SENSOR_UNIT_OUTPUT_NAME', SENSOR_UNIT_OUTPUT_NAME);
        setenv('SENSOR_POSITION_OUTPUT_NAME',SENSOR_POSITION_OUTPUT_NAME);
        setenv('STANDARD',STANDARD_UNIT);
        setenv('BOUT',STANDARD_BOUT);
        setenv('PLOT', BLOCK_PLOT);
        
        %% [2] Gait sequence detection
        
        if isfile([DirectoryPathSubjectGait, '\GSD_Output.mat'])
            
        else
            
            close all;
            
            bool_dummy_gsd = true;
            
            % gsd_algorithms: {'Hickey-original', 'Hickey-improved_th',...
            %     'TA_Wavelets', 'TA_Wavelets_v2', ...
            %     'TA_Iluz-original', 'TA_Iluz-improved_th',...
            %     'EPFL_V1-original', 'EPFL_V1-improved_th', ...
            %     'EPFL_V2-original', 'EPFL_V2-improved', ...
            %     'Rai', 'UNIBO', ...
            %     'Kheirkhahan' , 'Kheirkhahan_improved-th', 'GSD_GaitPy', 'Dummy' }
            
            if bool_dummy_gsd
                % Run GSD_dummy
                cd(fullfile('H:\Mobilise-D-TVS-Recommended-Algorithms\TD_ElGohary\gsd_dummy'));
                driver(indir, outdir)
            else
                % Run GSD_all
                setenv('ALGORITHM', 'EPFL_V1-improved_th'); % Choose algorithm
                cd(fullfile(librarydir, 'gsd_all'));
                driver(indir, outdir);
            end
            
            cd(currentwd);
            
        end %end GSD exist
        
        
        
        %% [4] Turning

            bool_dummy_TD = false;
            
            % Only one algorithm available (ElGohary)
            close all;
            
            if bool_dummy_TD
                cd(fullfile(librarydir, 'turning_dummy'));
                driver(indir, outdir)
            else
                cd(fullfile(librarydir, 'TD_ElGohary'));
                driver(indir, outdir)
            end
        
        cd(currentwd);        
        
    end %iSubject
    
end %iCohort
