%% =========================================================================
%                     BAMBI Kinematic Analysis – Max Activity Window
% =========================================================================
% Objective:
% This script processes 3D motion capture data (.c3d) for each participant 
% ("Bambi") by concatenating all their trials, and identifies the time window 
% (rounded down to the nearest 5 seconds) where the **most intense movement** 
% occurs.
%
% This moment of highest activity is determined by computing the Area Under 
% the Curve (AUC) of the foot speed over time within a sliding window, and 
% selecting the window with the maximum AUC. This window represents the 
% period with the greatest accumulated movement.
%
% Key Steps:
% - Load all trials for each participant
% - Extract and preprocess foot markers (fill gaps, filter)
% - Concatenate all speed signals for a participant
% - Define a common window size (based on shortest trial duration among participants)
% - Slide this window over the full speed signal to compute AUC
% - Identify and save the window with maximum AUC for each participant
% - Output diagnostic plots and an Excel summary of the results
%
% Outputs:
% - One plot per participant showing their full movement speed and the 
%   selected high-activity window (shaded)
% - An Excel table with the start/end frame of the max-AUC window for each Bambi
% =========================================================================

clear all
close all
clc
addpath('function')
% Paths
base_path = "S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\3 months_preparations ESMAC 2025\Used for presentation\3 months_TD_c3dfiles_v2";
outcome_path = "S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\OutcomeRawTD_ESMAC";
% Create Outcome folder if it doesn't exist
if ~exist(outcome_path, 'dir')
    mkdir(outcome_path);
end

% Load CSV data
csv_path = "S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\3 months_validity and reliability\3 months_validity and reliability.csv";
csv_tab = readtable(csv_path);

bambiID_list = csv_tab.InclusionNumber;

% Initialize structure to store all results
results = struct();

%% === Process each Bambi based on folder ===
for bambi = 1:length(bambiID_list)
    bambiID = bambiID_list(bambi);
    bambiID = bambiID{1};



    % Create output folder for each Bambi if it doesn't exist
    outcome_folder = fullfile(outcome_path, bambiID);
    if ~exist(outcome_folder, 'dir')
        mkdir(outcome_folder);
    end

    % Find all C3D files matching the current Bambi ID
    c3d_filenames = dir(fullfile(base_path, [bambiID, '*.c3d']));

    % ➤ Check if any .c3d files are found; if not, skip this Bambi
    if isempty(c3d_filenames)
        fprintf('No C3D files found for %s. Skipping.\n', bambiID);
        continue;
    end

    bambi_data = struct();

    % Initialize structure to store all marker data for this Bambi
    all_markers_data = [];

    %% === Process each C3D file ===
    for j = 1:length(c3d_filenames)
        marker_to_analyze = {"LANK", "LKNE", "LPEL", "RANK", "RKNE", "RPEL"};

        currentFile = fullfile(base_path, c3d_filenames(j).name);   
        fprintf('          File %s in process\n', c3d_filenames(j).name);

        % Get laterality (right/left) for the Bambi from the CSV
        numeroID = regexp(c3d_filenames(j).name, 'BAMBI0(\d{2})', 'tokens'); 
        numeroID = str2double(numeroID{1});  

        indices = find(strcmp(csv_tab.InclusionNumber, bambiID));
        laterality = csv_tab.Side(indices(1));

        if laterality == "right"
            side_to_analyze = "RANK";
        else 
            side_to_analyze = "LANK";
        end


        % Read motion capture data
        acq = btkReadAcquisition(currentFile{1});
        M = btkGetMarkers(acq);        

        % Extract markers of interest (without filling missing data)
        M_no_gap_filling = struct();
        for m = 1:length(marker_to_analyze)
            marker_name = marker_to_analyze{m};
            M_no_gap_filling.(marker_name) = M.(marker_name);
        end
        markers.name = fieldnames(M_no_gap_filling);

        % Reshape marker data for gap detection
        markerData = [];
        for m = 1:length(markers.name)
            for tt = 1:3
                markerData(end+1,:) = M_no_gap_filling.(markers.name{m})(:,tt);
            end
        end

        % Replace frames that are all zero with NaN
        framesWithGaps = all(markerData == 0, 1);
        markerData(:, framesWithGaps) = NaN;

        % Fill missing data using custom prediction function
        GapfilledDataSet = PredictMissingMarkers(markerData');
        GapfilledDataSet = GapfilledDataSet';

        % Rebuild the marker structure from filled data
        M_gap_filling = struct();
        r = 0;
        for m = 1:length(markers.name)
            for tt = 1:3
                r = r + 1;
                M_gap_filling.(markers.name{m})(:,tt) = GapfilledDataSet(r,:);
            end
        end       

        % Filter the marker data using a Butterworth low-pass filter
        n = btkGetPointFrameNumber(acq);
        frameRate = btkGetPointFrequency(acq);
        [butter_B, butter_A] = butter(4, 6/(frameRate/2), 'low');

        M_gap_filling_filter = struct();
        for m = 1:length(markers.name)
            M_gap_filling_filter.(markers.name{m}) = ...
                filtfilt(butter_B, butter_A, M_gap_filling.(markers.name{m}));
        end

        % Time-related info
        N = size(M_gap_filling_filter.(side_to_analyze), 1);  % Number of frames
        time = (0:N-1) / frameRate;
        duration = N / frameRate;
        disp(['N: ', num2str(N), ', Duration: ', num2str(duration), ', frameRate: ', num2str(frameRate)]);

        % Store data for this trial
        trial_data.time = time;
        trial_data.duration = duration;
        trial_data.M_no_gap_filling = M_no_gap_filling;
        trial_data.M_gap_filling_filter = M_gap_filling_filter;
        trial_data.side_to_analyze = side_to_analyze;
        trial_data.frameRate = frameRate;

        % Save trial to participant data
        bambi_data.(erase(c3d_filenames(j).name, '.c3d')) = trial_data;
    end  

    % Store participant data
    results.(bambiID) = bambi_data;

    % Save the full results structure (incrementally)
    % outcome_result = fullfile(outcome_path, 'resultats_time.mat');
    % save(outcome_result, 'results', '-v7.3');

    % Clear temporary variables for the next iteration
    clear markerData GapfilledDataSet markers M_no_gap_filling ...
          M_gap_filling M_gap_filling_filter trial_data acq
end

%% Summarize durations per participant
bambiIDs = fieldnames(results);
total_durations = struct();

for i = 1:length(bambiIDs)
    bambiID = bambiIDs{i};
    bambi_trials = results.(bambiID);
    trial_names = fieldnames(bambi_trials);
    total_duration = 0;

    for j = 1:length(trial_names)
        trial = bambi_trials.(trial_names{j});
        if isfield(trial, 'duration')
            total_duration = total_duration + trial.duration;
        else
            warning("Missing 'duration' in %s -> %s", bambiID, trial_names{j});
        end
    end

    total_durations.(bambiID) = total_duration;
end

disp(total_durations);

%% Create folder for time-window selection plots
time_selection_path = fullfile(outcome_path, 'Time selection');
if ~exist(time_selection_path, 'dir')
    mkdir(time_selection_path);
end

%% Calculate sliding AUC for each participant and plot results
AUC_windows = [];
[min_duration, ~] = min(struct2array(total_durations));
rounded_min_duration = floor(min_duration / 5) * 5;
window_duration = rounded_min_duration;

for i = 1:length(bambiIDs)
    bambiID = bambiIDs{i};
    bambi_trials = results.(bambiID);
    disp(bambiID)
    
    all_speeds_bambi = [];

    % Gather all speed data across trials
    trial_names = fieldnames(bambi_trials);
    for j = 1:length(trial_names)
        trial = bambi_trials.(trial_names{j});
        frameRate = trial.frameRate;
        window_frames = round(window_duration * frameRate); 

        marker_name = trial.side_to_analyze;
        marker_data = trial.M_gap_filling_filter.(marker_name);
        
        % Compute speed (magnitude of velocity)
        dx = diff(marker_data(:,1));
        dy = diff(marker_data(:,2));
        dz = diff(marker_data(:,3));
        speed = sqrt(dx.^2 + dy.^2 + dz.^2) / 1000 * frameRate;

        all_speeds_bambi = [all_speeds_bambi; speed];
    end

    % Compute sliding-window AUC
    AUC_all_bambi = [];
    for k = 1:(length(all_speeds_bambi) - window_frames)
        window_speed = all_speeds_bambi(k:k + window_frames - 1);
        AUC_all_bambi = [AUC_all_bambi; trapz(window_speed)];
    end

    [max_AUC, max_AUC_idx] = max(AUC_all_bambi);
    window_start = max_AUC_idx;
    window_end = max_AUC_idx + window_frames - 1;

    AUC_windows = [AUC_windows; {bambiID, window_start, window_end, max_AUC}];

    h1 = plot(all_speeds_bambi, 'b', 'LineWidth', 1.5);
    hold on;
    
    legend_handles = h1;
    legend_labels = {'Speed'};
    
    if window_end > window_start
        % Draw AUC window using a patch instead of rectangle
        x_rect = [window_start, window_end, window_end, window_start];
        y_min = min(all_speeds_bambi);
        y_max = max(all_speeds_bambi);
        y_rect = [y_min, y_min, y_max, y_max];
    
        h2 = patch(x_rect, y_rect, [0.5, 0.5, 0.5], ...
            'EdgeColor', 'none', 'FaceAlpha', 0.5);
    
        legend_handles(end+1) = h2;
        legend_labels{end+1} = 'Max AUC Window';
    end
    
    legend(legend_handles, legend_labels, 'Location', 'Best');


    ylim([0, 10]);

    saveas(gcf, fullfile(time_selection_path, [bambiID '_window_AUC_plot.png']));
    close(gcf);

    disp(['Bambi ', bambiID, ': Max AUC window from frame ', ...
        num2str(window_start), ' to ', num2str(window_end)]);
end

%% Save AUC results
AUC_table = cell2table(AUC_windows, ...
    'VariableNames', {'BambiID', 'WindowStart', 'WindowEnd', 'MaxAUC'});

writetable(AUC_table, fullfile(outcome_path, 'AUC_windows_results.xlsx'));