function TD = TD_ElGohary(imu, GSD, sensor_string)
plot_results  = strcmp(getenv_string('PLOT','false'), 'true');
fprintf('Plot: %s\n', mat2str(plot_results));

% Parameters from El-Gohary et al.,2014 (https://doi.org/10.3390/s140100356)
fc = 1.5;
turnThres = 45;

fs = imu.LowerBack.fs.Gyr;

% extract the GSD data
gs = GSD.LowerBack.GSD;

% only proceed if there have been any gait sequences detected
if isstruct(gs) && ~isempty(fieldnames(gs))
    % create an empty struct for the output
    TD = struct();
    
    % iterate over the gait sequences in the trial
    for k = 1:length(gs)
        current_gs = gs(k);
        
        % only extract the signal of the detected gait sequences
        gs_start_sample = round(current_gs.Start*fs)+1; % add 1 to avoid use of sample 0 which is not allowed in Matlab
        gs_end_sample = min(floor(current_gs.End*fs), length(imu.LowerBack.acc));

        % preprocessing of imu data
        imu_preprocessed =  preprocessing_standardized(imu, gs_start_sample, gs_end_sample,sensor_string);
        try
            % apply the turning detection algorithm
            [Turns]= turning_detection(imu_preprocessed, fc, fs, turnThres);
            
            % check current length of TD struct and determine index for appending
            if isempty(fieldnames(TD))
                start_idx = 0;
            else
                start_idx = length(TD);
            end
            for i=1:length(Turns.Magnitude)
                % reformat the turning result to Mobilise-D conventions
                TD(i+start_idx).Turn_Start = (Turns.TurnDur(i,1) + gs_start_sample) / fs;
                TD(i+start_idx).Turn_End = (Turns.TurnDur(i,2) + gs_start_sample) / fs;
                TD(i+start_idx).Turn_Duration = TD(i+start_idx).Turn_End - TD(i+start_idx).Turn_Start;
                TD(i+start_idx).Turn_Angle = Turns.Magnitude(i);
                TD(i+start_idx).Turn_AngularVelocity = imu_preprocessed(Turns.TurnDur(i,1):Turns.TurnDur(i,2));
                [~, max_idx] = max(abs(TD(i+start_idx).Turn_AngularVelocity));
                TD(i+start_idx).Turn_PeakAngularVelocity = TD(i+start_idx).Turn_AngularVelocity(max_idx);
                TD(i+start_idx).Turn_MeanAngularVelocity = mean(TD(i+start_idx).Turn_AngularVelocity);
                TD(i+start_idx).Turning_SharpTurn_Flag = [];
            end

            % Plot vertical lines for start (green) and end (red)
            % of turning phases
            if plot_results
                font_size = 14;
                figure; 
                set(gcf,'Position',[100 100 1200 400])
                subplot(121)
                plot(imu_preprocessed, 'LineWidth', 2); 
                hold on
                ylabel('Angular rate [deg/s]','FontSize', font_size);
                xlabel('Time [samples]','FontSize', font_size);
                if  ~isempty(fieldnames(TD))
                    td_boolean = zeros(size(imu_preprocessed(:,1)));
                    for j = 1:length(Turns.Magnitude)
                        current_turn = TD(j+start_idx);
                        td_boolean(round(current_turn.Turn_Start*fs-gs_start_sample):round(current_turn.Turn_End*fs-gs_start_sample)) = 200;
                    end
                    plot(td_boolean,'r','LineWidth',2)
                end
                lgd = legend('Gyr_V','Turn');
                lgd.FontSize=12;
                lgd.Location = 'southwest';
                ax=gca;
                ax.FontSize = 16;
                
                subplot(122)
                ylabel('Integrated Yaw Angle [deg]','FontSize', font_size);
                xlabel('Time [samples]','FontSize', font_size);
                hold on
                yaw = Turns.Yaw;
                plot(yaw, 'LineWidth',2)
                if  ~isempty(fieldnames(TD))
                    turn_yaw = yaw;
                    turn_yaw(~td_boolean) = NaN;
                    plot(turn_yaw,'r','LineWidth',2)
                end
                ax=gca;
                ax.FontSize = 16;
            end

        catch e
            warning('TD algorithm did not run successfully. Writing empty result to output. error identifier: %s, error messsage: %s Error in function %s() at line %d.', ...
            e.identifier, e.message, e.stack(1).name, e.stack(1).line);
            TD = [];
        end
    end
else
    % In case there was no gait sequence, return an emptpy result
    TD = [];
end
end
