%% =========================================================================
%                     BAMBI Kinematic Analysis – Max Activity Window
% =========================================================================
% Objective:
% This script processes 3D motion capture data (.c3d) for each participant 
% ("Bambi") by concatenating all their trials, and select a window
% previoulsy compute and stock in a .xlsx file
%
%
% Key Steps:
% - Load all trials for each participant
% - Extract and preprocess (fill gaps, filter)
% - Concatenate all marker trajectory
% - Apply the window 
% - Compute hip adduction/abduction based on a projection of the ankle
%  marker in the body plane
% - Compute hip flexion/extension based on a projection of the ankle
%  marker in the body plane
% - Compute and plot CDF PDF et KDE for both
% - Compute ankle velocity across time with a filtering
%
% Outputs:
% - Raw marker trajectories
% - Distance traveled by ankle
% - Hip joint angles (flexion and adduction)
% - Descriptive statistics of joint angles
% - Kernel Density Estimation and mode of joint angles
% - Ankle velocity profile and AUC
% - Final ankle position trace
% =========================================================================

clear all
close all
clc
addpath('function')
plot_fig=1;
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');

%% === Define paths ===
base_path = 'S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\3 months_preparations ESMAC 2025';
outcome_path = 'S:\KLab\#SHARE\RESEARCH\BAMBI\Outcome';

% Load previously calculated AUC window results
AUC_results_file = fullfile(outcome_path, 'AUC_windows_results.xlsx');
AUC_table = readtable(AUC_results_file);

% List of all C3D files available
files = dir(fullfile(base_path, '*.c3d'));

% Structure to store all participant results
all_bambi_results = struct();

%% === Process each Bambi based on AUC table ===
for i = 1:height(AUC_table)
    bambiID = AUC_table.BambiID{i};
    disp([bambiID, ' in process'])

    % Window of interest based on AUC result
    windowStart = AUC_table.WindowStart(i);
    windowEnd = AUC_table.WindowEnd(i);

    % Create output folder for each Bambi if it doesn't exist
    outcome_folder = fullfile(outcome_path, bambiID);
    if ~exist(outcome_folder, 'dir')
        mkdir(outcome_folder);
    end

    % Find all C3D files matching the current Bambi ID
    c3d_filenames = dir(fullfile(base_path, [bambiID, '*.c3d']));

    % Initialize structure to store all marker data for this Bambi
    all_markers_data = [];

    %% === Process each C3D file ===
    for j = 1:length(c3d_filenames)
        currentFile = fullfile(base_path, c3d_filenames(j).name);   
        fprintf('          File %s in process\n', c3d_filenames(j).name);

        % Get laterality (right/left) for the Bambi from the CSV
        numeroID = regexp(c3d_filenames(j).name, 'BAMBI0(\d{2})', 'tokens'); 
        numeroID = str2double(numeroID{1});  

        csv_name = '3 months in combination US & QTM_60s.csv'; 
        csv_path = fullfile(base_path, csv_name);
        csv_tab = readtable(csv_path);
        indices = find(csv_tab.InclusionNumber == numeroID);
        laterality = csv_tab.Side(indices(1));
        individual_data.laterality = laterality;

        % Read motion capture data
        acq = btkReadAcquisition(currentFile);
        M = btkGetMarkers(acq);  % Raw marker data
        markers.name = fieldnames(M);

        % Define expected marker categories
        categories = {'full', 'no_knee', 'no_upperlimb'};
        expected_markers.full = {'LANK','LKNE','LPEL','LSHO','LELB','LWRA','RANK','RKNE','RPEL','RSHO','RELB','RWRA'};
        expected_markers.no_knee = {'LANK','LPEL','LSHO','LWRA','RANK','RPEL','RSHO','RWRA'};
        expected_markers.no_upperlimb = {'LANK','LKNE','LPEL','LSHO','RANK','RKNE','RPEL','RSHO','RELB','RWRA'};

        % Determine marker category
        category = check_category(markers, expected_markers);
        individual_data.marker_category = category;

        % Display any missing markers
        if ~strcmp(category, 'unknown')
            missing_markers = setdiff(expected_markers.(category), markers.name);
            if ~isempty(missing_markers)
                disp('Missing markers:');
                disp(missing_markers);
            end
        end

        markers.distance = {'LANK','RANK'};  % Markers used for distance calculation

        % Get acquisition framerate
        frameRate = btkGetPointFrequency(acq);

        % Design low-pass Butterworth filter
        [butter_B, butter_A] = butter(4, 6/(frameRate/2), 'low');

        %% === Handle Missing Data ===
        % Convert marker structure to a 2D matrix
        markerData = [];
        r = 0;
        for m = 1:length(markers.name)
            for tt = 1:3
                r = r + 1;
                markerData(r,:) = M.(markers.name{m})(:,tt);
            end
        end

        % Set zeros to NaN
        markerData(markerData == 0) = NaN;

        % Predict/fill missing data
        GapfilledDataSet = PredictMissingMarkers(markerData')';
        
        % Rebuild marker structure from gap-filled data
        M_f = struct();
        r = 0;
        for m = 1:length(markers.name)
            for tt = 1:3
                r = r + 1;
                M_f.(markers.name{m})(:,tt) = GapfilledDataSet(r,:);
            end
        end

        % Apply filtering
        for m = 1:length(markers.name)
            try
                M_f.(markers.name{m}) = filtfilt(butter_B, butter_A, M_f.(markers.name{m}));
            catch ME
                disp(['Error filtering marker: ', markers.name{m}]);
                disp(['Message: ', ME.message]);
                disp(ME.stack);
            end
        end

        % Store this file's processed marker data
        all_markers_data = [all_markers_data; M_f];
    end

    %% === Concatenate Marker Data Across All Files ===
    combined_markers_data = struct();
    marker_fields = fieldnames(all_markers_data(1));  % Marker names

    for marker_in_list = 1:length(marker_fields)
        marker = marker_fields{marker_in_list};
        concatenated_data = [];

        for trials_to_stock = 1:length(all_markers_data)
            if isfield(all_markers_data(trials_to_stock), marker)
                concatenated_data = [concatenated_data; ...
                                     all_markers_data(trials_to_stock).(marker)];
            end
        end

        % Store combined marker data
        combined_markers_data.(marker) = concatenated_data;
    end
    disp([bambiID, ' data were correctly concatened'])

    %% === Kinematic Processing in Thorax Local Coordinate System ===
    
    n = length(combined_markers_data.RANK);  % Number of frames
    
    % Define virtual markers (midpoints)
    midShoulder  = (combined_markers_data.LSHO + combined_markers_data.RSHO) / 2;
    midLPelSho   = (combined_markers_data.LSHO + combined_markers_data.LPEL) / 2;
    midRPelSho   = (combined_markers_data.RSHO + combined_markers_data.RPEL) / 2;
    midPelvis    = (combined_markers_data.LPEL + combined_markers_data.RPEL) / 2;
    center       = (midShoulder + midPelvis) / 2;
    
    % Define thorax local coordinate system (LCS)
    xtmp = (midLPelSho - midRPelSho);                          % Temporary medio-lateral axis
    y_gt = f_t_Vnorm(midShoulder - midPelvis);                 % Y (vertical) axis
    z_gt = f_t_Vnorm(f_t_cross(xtmp, y_gt));                   % Z (antero-posterior) axis
    x_gt = f_t_Vnorm(f_t_cross(y_gt, z_gt));                   % X (medio-lateral) axis
    
    % Reshape for matrix multiplication
    O_gt = permute(center, [2 3 1]);                           
    R_gt = [permute(x_gt, [2 3 1]), permute(y_gt, [2 3 1]), permute(z_gt, [2 3 1])];
    T_gt = [R_gt O_gt; zeros(1,3,n) ones(1,1,n)];              
    
    % Transform all markers into thorax LCS
    list_marker = fieldnames(combined_markers_data);   
    n_marker = length(list_marker);                        
    
    for j = 1:n_marker
        % Prepare marker for transformation (homogeneous coordinates)
        tmp1 = [permute(combined_markers_data.(list_marker{j}), [2 3 1]); ones(1,1,n)];
        tmp2 = Mprod_array3(Tinv_array3(T_gt), tmp1);           % Apply inverse transform
        m_th.(list_marker{j}) = permute(tmp2(1:3,1,:), [3 1 2]);        % Save transformed marker
        Marker_in_pelvis_frame.(list_marker{j}) = m_th.(list_marker{j});        % For plotting/export
    end
    
    % Prepare markers for 2D plotting (XY plane)
    for j = 1:length(markers.name)
        Marker_in_pelvis_frame_plot.X(:,j) = Marker_in_pelvis_frame.(markers.name{j})(:,1);
        Marker_in_pelvis_frame_plot.Y(:,j) = Marker_in_pelvis_frame.(markers.name{j})(:,2);
    end
    
    % Compute total ankle travel distance (LANK and RANK)
    for m = 1:length(markers.distance)
        d = 0;
        for t = 2:length(combined_markers_data.(markers.distance{m}))
            d = d + norm(combined_markers_data.(markers.distance{m})(t,:) - combined_markers_data.(markers.distance{m})(t-1,:));
        end
        Outcome.(markers.distance{m})(i).Distance = d;  
    end
    
    %% === Optional 2D Segment Plotting ===
    figure;
    axis equal
    hold on
    plot(Marker_in_pelvis_frame_plot.X, Marker_in_pelvis_frame_plot.Y, '.');  % 2D scatter of all markers

    % Draw segments based on marker availability category
    switch category
        case 'full'
            draw_segment(Marker_in_pelvis_frame.LANK, Marker_in_pelvis_frame.LKNE);
            draw_segment(Marker_in_pelvis_frame.LKNE, Marker_in_pelvis_frame.LPEL);
            draw_segment(Marker_in_pelvis_frame.LPEL, Marker_in_pelvis_frame.LSHO);
            draw_segment(Marker_in_pelvis_frame.LSHO, Marker_in_pelvis_frame.LELB);
            draw_segment(Marker_in_pelvis_frame.LELB, Marker_in_pelvis_frame.LWRA);
            draw_segment(Marker_in_pelvis_frame.LSHO, Marker_in_pelvis_frame.RSHO);
            draw_segment(Marker_in_pelvis_frame.LPEL, Marker_in_pelvis_frame.RPEL);
            draw_segment(Marker_in_pelvis_frame.RANK, Marker_in_pelvis_frame.RKNE);
            draw_segment(Marker_in_pelvis_frame.RKNE, Marker_in_pelvis_frame.RPEL);
            draw_segment(Marker_in_pelvis_frame.RPEL, Marker_in_pelvis_frame.RSHO);
            draw_segment(Marker_in_pelvis_frame.RSHO, Marker_in_pelvis_frame.RELB);
            draw_segment(Marker_in_pelvis_frame.RELB, Marker_in_pelvis_frame.RWRA);

        case 'no_knee'
            draw_segment(Marker_in_pelvis_frame.LPEL, Marker_in_pelvis_frame.LSHO);
            draw_segment(Marker_in_pelvis_frame.LSHO, Marker_in_pelvis_frame.RSHO);
            draw_segment(Marker_in_pelvis_frame.LPEL, Marker_in_pelvis_frame.RPEL);
            draw_segment(Marker_in_pelvis_frame.RPEL, Marker_in_pelvis_frame.RSHO);

        case 'no_upperlimb'
            draw_segment(Marker_in_pelvis_frame.LANK, Marker_in_pelvis_frame.LKNE);
            draw_segment(Marker_in_pelvis_frame.LKNE, Marker_in_pelvis_frame.LPEL);
            draw_segment(Marker_in_pelvis_frame.LPEL, Marker_in_pelvis_frame.LSHO);
            draw_segment(Marker_in_pelvis_frame.LSHO, Marker_in_pelvis_frame.RSHO);
            draw_segment(Marker_in_pelvis_frame.LPEL, Marker_in_pelvis_frame.RPEL);
            draw_segment(Marker_in_pelvis_frame.RANK, Marker_in_pelvis_frame.RKNE);
            draw_segment(Marker_in_pelvis_frame.RKNE, Marker_in_pelvis_frame.RPEL);
            draw_segment(Marker_in_pelvis_frame.RPEL, Marker_in_pelvis_frame.RSHO);
    end

    % Save plot
    saveas(gcf, fullfile(outcome_folder, [category '_2d_plot.png']));
    close(gcf);

    %% === Distance Between LANK and RANK Over Time ===
    N = size(combined_markers_data.LANK, 1);     % Number of time frames
    time = (0:N-1) / frameRate;                  % Time vector in seconds
    individual_data.time_duration = time;        % Store time duration
    
    % Compute Euclidean distance frame-by-frame
    distances = zeros(1, N);
    for i = 1:N
        distances(i) = norm(combined_markers_data.LANK(i,:) - combined_markers_data.RANK(i,:));
    end
    individual_data.distance_RL_ANK = distances;
    
    % Plot and save
    figure;
    plot(time, distances, 'b-', 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Distance (m)');
    title('Evolution of Distance Between LANK and RANK');
    grid on;
    saveas(gcf, fullfile(outcome_folder, 'Distance_between_RANK_LANK.png'));
    close(gcf);
      
    % Calculate midpoints of shoulders and pelvis for all frames
    midShoulder = (combined_markers_data.LSHO + combined_markers_data.RSHO) / 2;
    midPelvis = (combined_markers_data.LPEL + combined_markers_data.RPEL) / 2;

    %% === Compute hip abduction/adduction angles ===

    % Initialize arrays to store hip abduction/adduction angles 
    hip_angle_r_deg_array_add = zeros(1, N);
    hip_angle_l_deg_array_add = zeros(1, N);
    
    for i = 1:N
        % Vectors from mid-shoulder to pelvis markers (right and left)
        mid_shoulder_hip_r = combined_markers_data.RPEL(i,:) - midShoulder(i,:);
        mid_shoulder_hip_l = combined_markers_data.LPEL(i,:) - midShoulder(i,:);
        
        % Normal vector to the trunk plane (defined by pelvis and shoulders)
        normal_vector_plan = cross(mid_shoulder_hip_r, mid_shoulder_hip_l);
    
        % Reference vectors along the pelvis axis (right to left and vice versa)
        vector_ref_r = combined_markers_data.RPEL(i,:) - combined_markers_data.LPEL(i,:);
        vector_ref_l = combined_markers_data.LPEL(i,:) - combined_markers_data.RPEL(i,:);
    
        % Thigh vectors (from pelvis to ankle)
        thigh_vector_R = combined_markers_data.RANK(i,:) - combined_markers_data.RPEL(i,:);
        thigh_vector_L = combined_markers_data.LANK(i,:) - combined_markers_data.LPEL(i,:);
    
        % Project thigh vectors onto the trunk plane
        thigh_vector_R_proj = thigh_vector_R - (dot(thigh_vector_R, normal_vector_plan) / dot(normal_vector_plan, normal_vector_plan)) * normal_vector_plan;
        thigh_vector_L_proj = thigh_vector_L - (dot(thigh_vector_L, normal_vector_plan) / dot(normal_vector_plan, normal_vector_plan)) * normal_vector_plan;
    
        % Compute cosine of the angle using dot product
        cos_hip_angle_r = dot(vector_ref_r, thigh_vector_R_proj) / (norm(vector_ref_r) * norm(thigh_vector_R_proj));
        cos_hip_angle_l = dot(vector_ref_l, thigh_vector_L_proj) / (norm(vector_ref_l) * norm(thigh_vector_L_proj));
    
        % Convert angle from radians to degrees and offset by 90°
        hip_angle_r_deg_array_add(i) = rad2deg(acos(cos_hip_angle_r)) - 90;
        hip_angle_l_deg_array_add(i) = rad2deg(acos(cos_hip_angle_l)) - 90;
    end
    
    %% === Plot hip angles over time ===
    figure; hold on;
    plot(time, hip_angle_r_deg_array_add, 'b-', 'LineWidth', 2);  % Right hip
    plot(time, hip_angle_l_deg_array_add, 'r-', 'LineWidth', 2);  % Left hip
    xlabel('Time (s)');
    ylabel('Hip Angle (degrees)');
    title('Hip Angles over Time body ref');
    legend('Right Hip Angle', 'Left Hip Angle');
    grid on;
    saveas(gcf, fullfile(outcome_folder, 'Hip angle add abd.png'));
    close(gcf);
    
    %% === Compute and plot CDF ===
    sorted_hip_r_add = sort(hip_angle_r_deg_array_add);
    sorted_hip_l_add = sort(hip_angle_l_deg_array_add);
    n = length(sorted_hip_r_add);
    cdf_values = (1:n) / n;
    
    figure; hold on;
    plot(sorted_hip_r_add, cdf_values, 'LineWidth', 2);
    plot(sorted_hip_l_add, cdf_values, 'LineWidth', 2);
    xlabel('Angle (°)');
    ylabel('CDF');
    title('Cumulative Distribution Function of Hip Angle (Left Leg)');
    legend('Right Hip Angle', 'Left Hip Angle');
    grid on;
    saveas(gcf, fullfile(outcome_folder, 'CDF of Hip Angle add abd.png'));
    close(gcf);
    
    %% === Compute and plot PDF with histogram + KDE ===
    % Define bin width
    tranches = 1;
    
    % For right side
    edges_r = min(sorted_hip_r_add):tranches:max(sorted_hip_r_add)+tranches;
    [counts_r, bin_centers_r] = hist(sorted_hip_r_add, edges_r);
    [~, idx_r] = max(counts_r);
    mode_r_add = bin_centers_r(idx_r);
    pdf_r = counts_r / sum(counts_r);
    
    % For left side
    edges_l = min(sorted_hip_l_add):tranches:max(sorted_hip_l_add)+tranches;
    [counts_l, bin_centers_l] = hist(sorted_hip_l_add, edges_l);
    [~, idx_l] = max(counts_l);
    mode_l_add = bin_centers_l(idx_l);
    pdf_l = counts_l / sum(counts_l);
    
    % Create subplot for both sides
    figure;
    
    % Right side plot
    subplot(1, 2, 1);
    bar(bin_centers_r, pdf_r, 'histc'); hold on;
    [f_r_add, xi_r_add] = ksdensity(sorted_hip_r_add, 'Bandwidth', 2);
    plot(xi_r_add, f_r_add, 'r-', 'LineWidth', 2);
    xlabel('Angle (degrees)');
    ylabel('Probability Density');
    title('Right Side: PDF and Smoothed Curve');
    legend('Histogram', 'KDE Curve');
    grid on;
    
    % Left side plot
    subplot(1, 2, 2);
    bar(bin_centers_l, pdf_l, 'histc'); hold on;
    [f_l_add, xi_l_add] = ksdensity(sorted_hip_l_add, 'Bandwidth', 2);
    plot(xi_l_add, f_l_add, 'b-', 'LineWidth', 2);
    xlabel('Angle (degrees)');
    ylabel('Probability Density');
    title('Left Side: PDF and Smoothed Curve');
    legend('Histogram', 'KDE Curve');
    grid on;
    
    saveas(gcf, fullfile(outcome_folder, 'PDF of Hip Angle add abd.png'));
    close(gcf);

    %% === Compute hip flexion/extension angles ===

    % Initialize arrays to store hip flexion/extension angles
    hip_angle_r_deg_array_flex = zeros(1, N);
    hip_angle_l_deg_array_flex = zeros(1, N);
    
    for i = 1:N
        % Normal vector to the plane defined by right and left pelvis
        normal_vector_plan = combined_markers_data.LPEL(i,:) - combined_markers_data.RPEL(i,:);
        
        % Reference vector (from mid pelvis to mid shoulder)
        vector_ref = midPelvis(i,:) - midShoulder(i,:); 
        
        % Thigh vectors for right and left legs
        thigh_vector_R = combined_markers_data.RANK(i,:) - combined_markers_data.RPEL(i,:);
        thigh_vector_L = combined_markers_data.LANK(i,:) - combined_markers_data.LPEL(i,:);
    
        % Project thigh vectors onto the trunk plane
        thigh_vector_R_proj = thigh_vector_R - (dot(thigh_vector_R, normal_vector_plan) / dot(normal_vector_plan, normal_vector_plan)) * normal_vector_plan;
        thigh_vector_L_proj = thigh_vector_L - (dot(thigh_vector_L, normal_vector_plan) / dot(normal_vector_plan, normal_vector_plan)) * normal_vector_plan;
    
        % Calculate the cosine of the angle using the dot product
        cos_hip_angle_r = dot(vector_ref, thigh_vector_R_proj) / (norm(vector_ref) * norm(thigh_vector_R_proj));
        cos_hip_angle_l = dot(vector_ref, thigh_vector_L_proj) / (norm(vector_ref) * norm(thigh_vector_L_proj));
    
        % Calculate the angle in radians and convert to degrees
        hip_angle_r_rad = acos(cos_hip_angle_r);
        hip_angle_r_deg = rad2deg(hip_angle_r_rad);
       
        hip_angle_l_rad = acos(cos_hip_angle_l);
        hip_angle_l_deg = rad2deg(hip_angle_l_rad);
    
        % Store the angle for the current frame
        hip_angle_r_deg_array_flex(i) = hip_angle_r_deg;
        hip_angle_l_deg_array_flex(i) = hip_angle_l_deg;
    end    
    
    %% === Plot hip flexion/extension angles over time ===
    % Plot the hip angles over time
    figure;
    hold on; % Keep both plots in the same figure
    
    % Plot the right hip angle
    plot(time, hip_angle_r_deg_array_flex, 'b-', 'LineWidth', 2);
    
    % Plot the left hip angle
    plot(time, hip_angle_l_deg_array_flex, 'r-', 'LineWidth', 2);
    
    % Labels and title
    xlabel('Time (s)');
    ylabel('Hip Angle (degrees)');
    title('Hip Angles over Time ground ref');
    grid on;
    
    % Add a legend
    legend('Right Hip Angle', 'Left Hip Angle');
    
    % Save the figure
    saveas(gcf, fullfile(outcome_folder, 'Hip angle flex ext.png'));
    close(gcf);

    %% === Compute and plot CDF ===
    sorted_hip_l_flex = sort(hip_angle_l_deg_array_flex);
    sorted_hip_r_flex = sort(hip_angle_r_deg_array_flex);
    n = length(sorted_hip_r_flex);
    cdf_values = (1:n) / n;
    
    figure; hold on;
    plot(sorted_hip_r_flex, cdf_values, 'LineWidth', 2);
    plot(sorted_hip_l_flex, cdf_values, 'LineWidth', 2);
    xlabel('Angle (°)');
    ylabel('CDF');
    title('Cumulative Distribution Function of Hip Angle (Left Leg)');
    legend('Right Hip Angle', 'Left Hip Angle');
    grid on;
    saveas(gcf, fullfile(outcome_folder, 'CDF of Hip angle flex ext.png'));
    close(gcf);
    
%% === Compute and plot PDF with histogram + KDE ===
    % Define bin width
    tranches = 1;
    
    % For right side
    edges_r = min(sorted_hip_r_flex):tranches:max(sorted_hip_r_flex)+tranches;
    [counts_r, bin_centers_r] = hist(sorted_hip_r_flex, edges_r);
    [~, idx_r] = max(counts_r);
    mode_r_flex = bin_centers_r(idx_r);
    pdf_r = counts_r / sum(counts_r);
    
    % For left side
    edges_l = min(sorted_hip_l_flex):tranches:max(sorted_hip_l_flex)+tranches;
    [counts_l, bin_centers_l] = hist(sorted_hip_l_flex, edges_l);
    [~, idx_l] = max(counts_l);
    mode_l_flex = bin_centers_l(idx_l);
    pdf_l = counts_l / sum(counts_l);
    
    % Create subplot for both sides
    figure;
    
    % Right side plot
    subplot(1, 2, 1);
    bar(bin_centers_r, pdf_r, 'histc'); hold on;
    [f_r_flex, xi_r_flex] = ksdensity(sorted_hip_r_flex, 'Bandwidth', 2);
    plot(xi_r_flex, f_r_flex, 'r-', 'LineWidth', 2);
    xlabel('Angle (degrees)');
    ylabel('Probability Density');
    title('Right Side: PDF and Smoothed Curve');
    legend('Histogram', 'KDE Curve');
    grid on;
    
    % Left side plot
    subplot(1, 2, 2);
    bar(bin_centers_l, pdf_l, 'histc'); hold on;
    [f_l_flex, xi_l_flex] = ksdensity(sorted_hip_l_flex, 'Bandwidth', 2);
    plot(xi_l_flex, f_l_flex, 'b-', 'LineWidth', 2);
    xlabel('Angle (degrees)');
    ylabel('Probability Density');
    title('Left Side: PDF and Smoothed Curve');
    legend('Histogram', 'KDE Curve');
    grid on;
    
    saveas(gcf, fullfile(outcome_folder, 'PDF of Hip Angle flex ext.png'));
    close(gcf);

    %% === Compute and plot ankle velocity over time ===

    % Define the time interval between frames
    delta_t = 1 / frameRate;  
    
    % Initialize velocity vectors
    velocity_RANK = zeros(N-1, 1);  % Right ankle velocity
    velocity_LANK = zeros(N-1, 1);  % Left ankle velocity
    
    %% === Calculate the velocity for each frame ===

    for i = 1:N-1
        % Get the marker positions at the current and next frame
        pos_t_R = combined_markers_data.RANK(i, :);  
        pos_t1_R = combined_markers_data.RANK(i+1, :);  
    
        pos_t_L = combined_markers_data.LANK(i, :); 
        pos_t1_L = combined_markers_data.LANK(i+1, :);  
    
        % Calculate the distance between positions (in mm)
        distance_mm_R = sqrt(sum((pos_t1_R - pos_t_R).^2)); 
        distance_mm_L = sqrt(sum((pos_t1_L - pos_t_L).^2)); 
    
        % Convert distance to meters (from mm)
        distance_m_R = distance_mm_R / 1000;  
        distance_m_L = distance_mm_L / 1000;  
    
        % Calculate the velocity (in m/s): distance / delta_t
        velocity_RANK(i) = distance_m_R / delta_t;  
        velocity_LANK(i) = distance_m_L / delta_t; 
    end
    
    %% === Apply a Butterworth filter === maybe not the best because it give negative value

    % Parameters for Butterworth filter
    order = 4;  % Filter order
    cutoff_frequency = 4;  % Cutoff frequency in Hz
    
    % Normalize the cutoff frequency with respect to the frame rate
    [b, a] = butter(order, cutoff_frequency / (frameRate / 2), 'low');  
    
    % Apply the Butterworth filter on velocity data
    filtered_velocity_RANK = filtfilt(b, a, velocity_RANK);  
    filtered_velocity_LANK = filtfilt(b, a, velocity_LANK);  
    
    % Plot both raw and filtered velocities
    figure;
    plot(1:N-1, filtered_velocity_RANK, 'r', 'LineWidth', 2);  % Plot right ankle velocity
    hold on;
    plot(1:N-1, filtered_velocity_LANK, 'b', 'LineWidth', 2);  % Plot left ankle velocity
    
    xlabel('Frame');  % X-axis label
    ylabel('Velocity (m/s)');  % Y-axis label
    title('Marker Velocity with Butterworth Filter');  % Plot title
    legend('Right Ankle', 'Left Ankle');  % Legend
    grid on;  % Enable grid
    
    % Save the plot
    saveas(gcf, fullfile(outcome_folder, 'Ankle_velocity.png'));
    close(gcf);
    
    % Calculate the Area Under the Curve (AUC) for both ankles using the trapezoidal method
    AUC_RANK = trapz(filtered_velocity_RANK) * delta_t;  
    AUC_LANK = trapz(filtered_velocity_LANK) * delta_t;  

    %% === Save all outcomes ===
    individual_data.LANK = combined_markers_data.LANK;
    individual_data.RANK = combined_markers_data.RANK;
    individual_data.RKNE = combined_markers_data.RKNE;
    individual_data.LKNE = combined_markers_data.LKNE;
    individual_data.LPEL = combined_markers_data.LPEL;
    individual_data.RPEL = combined_markers_data.RPEL;
    individual_data.LSHO = combined_markers_data.LSHO;
    individual_data.RSHO = combined_markers_data.RSHO;

    individual_data.RELB = combined_markers_data.RELB;
    individual_data.LELB = combined_markers_data.LELB;
    individual_data.RWRA = combined_markers_data.RWRA;
    individual_data.LWRA = combined_markers_data.LWRA;

    if laterality == "right"
        individual_data.Distance_travel_ankle_mm = Outcome.RANK.Distance;

        individual_data.hip_angle_add = hip_angle_r_deg_array_add;
        individual_data.mean_hip_angle_add = mean(hip_angle_r_deg_array_add);
        individual_data.std_hip_angle_add = std(hip_angle_r_deg_array_add);
        individual_data.skew_hip_angle_add = skewness(hip_angle_r_deg_array_add);
        individual_data.kurt_hip_angle_add = kurtosis(hip_angle_r_deg_array_add);
        individual_data.KDE_hip_angle_add = [f_r_add, xi_r_add];
        individual_data.mode_add = mode_r_add;

        individual_data.hip_angle_flex = hip_angle_r_deg_array_flex;
        individual_data.mean_hip_angle_flex = mean(hip_angle_r_deg_array_flex);
        individual_data.std_hip_angle_flex = std(hip_angle_r_deg_array_flex);
        individual_data.skew_hip_angle_flex = skewness(hip_angle_r_deg_array_flex);
        individual_data.kurt_hip_angle_flex = kurtosis(hip_angle_r_deg_array_flex);
        individual_data.KDE_hip_angle_flex = [f_r_flex, xi_r_flex];
        individual_data.mode_flex = mode_r_flex;

        individual_data.velocity_ankle = filtered_velocity_RANK;
        individual_data.AUC_velocity_ankle = AUC_RANK;
        individual_data.ankle_pos = combined_markers_data.RANK;

    else
        individual_data.Distance_travel_ankle_mm = Outcome.LANK.Distance;

        individual_data.hip_angle_add = hip_angle_l_deg_array_add;
        individual_data.mean_hip_angle_add = mean(hip_angle_l_deg_array_add);
        individual_data.std_hip_angle_add = std(hip_angle_l_deg_array_add);
        individual_data.skew_hip_angle_add = skewness(hip_angle_l_deg_array_add);
        individual_data.kurt_hip_angle_add = kurtosis(hip_angle_l_deg_array_add);
        individual_data.KDE_hip_angle_add_x = xi_r_add;
        individual_data.KDE_hip_angle_add_f = f_r_add;
        individual_data.mode_add = mode_l_add;

        individual_data.hip_angle_flex = hip_angle_l_deg_array_flex;
        individual_data.mean_hip_angle_flex = mean(hip_angle_l_deg_array_flex);
        individual_data.std_hip_angle_flex = std(hip_angle_l_deg_array_flex);
        individual_data.skew_hip_angle_flex = skewness(hip_angle_l_deg_array_flex);
        individual_data.kurt_hip_angle_flex = kurtosis(hip_angle_l_deg_array_flex);
        individual_data.KDE_hip_angle_flex_x = xi_l_flex;
        individual_data.KDE_hip_angle_flex_f = f_l_flex;
        individual_data.mode_flex = mode_l_flex;

        individual_data.velocity_ankle = filtered_velocity_LANK;
        individual_data.AUC_velocity_ankle = AUC_LANK;
        individual_data.ankle_pos = combined_markers_data.LANK;
    end

    results.(bambiID) = individual_data;

    outcome_result = fullfile(outcome_path, 'resultats.mat');
    save(outcome_result, 'results');
    
    clear markerData Marker_in_pelvis_frame Marker_in_pelvis_frame_plot M_f GapfilledDataSet individual_data
end
