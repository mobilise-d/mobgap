function GSD = build_gsd_output(GSD, WB_start, WB_end, algorithm_output_fs)

% in case of no detected initial contacts return SD as is
if isempty(WB_start)
    return
else
    % check current length of gs_refined and determine index for appending
    if isempty(fieldnames(GSD))
        start_idx = 0;
    else
        start_idx = length(GSD);
    end
    
    for ref_bout = 1:length(WB_start)
        
        % extract gait sequence start
        gs_start = WB_start(ref_bout);       
        % end of the gait sequence
        gs_end = WB_end(ref_bout);

        % save to the output struct
        GSD(ref_bout+start_idx).Start = gs_start;
        GSD(ref_bout+start_idx).End = gs_end;
        GSD(ref_bout+start_idx).GSD_fs = algorithm_output_fs;
    end
end
