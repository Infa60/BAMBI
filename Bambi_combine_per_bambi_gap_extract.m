%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BAMBI Kinematic Data Processing Script
%
% Description:
% This script processes 3D kinematic data from C3D motion capture files 
% collected for the BAMBI project. It performs marker-based preprocessing, 
% categorization based on marker availability, gap filling, filtering, 
% coordinate transformation into a thorax-based local frame, 2D visualization, 
% and storage of structured results for further analysis.
%
% Workflow Overview:
% 1. Load all available C3D files and identify unique participant IDs.
% 2. For each participant:
%    - Identify relevant C3D files.
%    - Extract metadata (e.g., laterality) from an associated CSV file.
%    - Read marker trajectories from C3D using BTK.
%    - Classify marker configuration into categories ('full', 'no_knee', etc.).
%    - Handle missing data (replace zeros with NaN, gap fill).
%    - Apply a low-pass Butterworth filter (6 Hz cutoff).
%    - Concatenate data across trials for each participant.
%    - Compute thorax-based local coordinate system.
%    - Transform marker coordinates into the local frame.
%    - Plot a 2D projection of the marker configuration (XY plane).
%    - Save figures and structured marker data.
%
% Inputs:
% - C3D motion capture files stored in a specified directory.
% - A CSV file containing participant metadata (laterality, inclusion numbers).
%
% Outputs:
% - A .mat file ('resultat_simple.mat') containing processed marker data
%   for each participant, categorized by marker configuration.
% - A 2D PNG figure of the marker layout per participant.
%
% Dependencies:
% - BTK toolbox for reading C3D files.
% - Custom functions: check_category, PredictMissingMarkers, draw_segment,
%   f_t_Vnorm, f_t_cross, Mprod_array3, Tinv_array3
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc
addpath('function')
plot_fig=1;
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');

%% === Define paths ===
outcome_path = 'S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\Outcome';

base_path = 'S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\3 months_validity and reliability';

% Liste tous les fichiers dans le dossier
files = dir(fullfile(base_path, '*')); 

% Initialise une cellule pour stocker les noms extraits
extractedNames = {};

% Parcours les fichiers
for i = 1:length(files)
    if ~files(i).isdir
        % Nom du fichier
        fileName = files(i).name;

        % Trouver la position du premier "_"
        underscoreIdx = strfind(fileName, '_');

        if ~isempty(underscoreIdx)
            % Extraire la partie avant le premier "_"
            namePart = fileName(1:underscoreIdx(1)-1);
        else
            % Si pas de "_", prendre le nom complet
            namePart = fileName;
        end

        % Ajouter à la liste
        extractedNames{end+1} = namePart;
    end
end

bambiID_list = unique(extractedNames');

% List of all C3D files available
files = dir(fullfile(base_path, '*.c3d'));

% Structure to store all participant results
all_bambi_results = struct();

%% === Process each Bambi based on AUC table ===
for bambi = 1:length(bambiID_list)
    bambi_data = struct();
    bambiID = bambiID_list(bambi);
    bambiID = bambiID{1};

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

        csv_path = "S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\3 months_validity and reliability.csv";
        csv_tab = readtable(csv_path);
        indices = find(strcmp(csv_tab.InclusionNumber, bambiID));
        laterality = csv_tab.Side(indices(1));
        individual_data.laterality = laterality;

        % Read motion capture data
        acq = btkReadAcquisition(currentFile);
        M = btkGetMarkers(acq);  % Raw marker data
        markers.name = fieldnames(M);

        % Define expected marker categories
        categories = {'full', 'no_knee', 'no_upperlimb'};
        expected_markers.full = {'CSHD','FSHD','LSHD','RSHD','LANK','LKNE','LPEL','LSHO','LELB','LWRA','RANK','RKNE','RPEL','RSHO','RELB','RWRA'};
        expected_markers.no_knee = {'LANK','LPEL','LSHO','LWRA','RANK','RPEL','RSHO','RWRA'};
        expected_markers.no_upperlimb = {'LANK','LKNE','LPEL','LSHO','RANK','RKNE','RPEL','RSHO','RELB','RWRA'};
        expected_markers.no_head = {'LANK','LKNE','LPEL','LSHO','LELB','LWRA','RANK','RKNE','RPEL','RSHO','RELB','RWRA'};

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
    
    %% === Time Duration ===
    n = size(combined_markers_data.RKNE, 1);     % Number of time frames
    time = (0:n-1) / frameRate;                  % Time vector in seconds
    individual_data.time_duration = time;        % Store time duration

    %% === Kinematic Processing in Thorax Local Coordinate System ===
    
        %% Restrict marker data to the window of interest
    % fields = fieldnames(combined_markers_data);
    % for f = 1:length(fields)
        % combined_markers_data.(fields{f}) = combined_markers_data.(fields{f})(windowStart:windowEnd, :);
    % end

    % % Update the number of frames based on the window
    % n = windowEnd - windowStart + 1;
        
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
    

    
    %% === Optional 2D Segment Plotting ===
    figure;
    axis equal
    hold on
    plot(Marker_in_pelvis_frame_plot.X, Marker_in_pelvis_frame_plot.Y, '.');  % 2D scatter of all markers

    % Draw segments based on marker availability category
    switch category
        case 'full'
            draw_segment(Marker_in_pelvis_frame.CSHD, Marker_in_pelvis_frame.CSHD);
            draw_segment(Marker_in_pelvis_frame.FSHD, Marker_in_pelvis_frame.FSHD);
            draw_segment(Marker_in_pelvis_frame.LSHD, Marker_in_pelvis_frame.LSHD);
            draw_segment(Marker_in_pelvis_frame.RSHD, Marker_in_pelvis_frame.RSHD);
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
        case 'no_head'
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
    end

    % Save plot
    saveas(gcf, fullfile(outcome_folder, [category '_2d_plot.png']));
    close(gcf);

    


    %% === Save all outcomes ===
    switch category
        case 'full'
            individual_data.LANK = Marker_in_pelvis_frame.LANK;
            individual_data.RANK = Marker_in_pelvis_frame.RANK;
            individual_data.RKNE = Marker_in_pelvis_frame.RKNE;
            individual_data.LKNE = Marker_in_pelvis_frame.LKNE;
            individual_data.LPEL = Marker_in_pelvis_frame.LPEL;
            individual_data.RPEL = Marker_in_pelvis_frame.RPEL;
            individual_data.LSHO = Marker_in_pelvis_frame.LSHO;
            individual_data.RSHO = Marker_in_pelvis_frame.RSHO;
        
            individual_data.RELB = Marker_in_pelvis_frame.RELB;
            individual_data.LELB = Marker_in_pelvis_frame.LELB;
            individual_data.RWRA = Marker_in_pelvis_frame.RWRA;
            individual_data.LWRA = Marker_in_pelvis_frame.LWRA;
        
            individual_data.CSHD = Marker_in_pelvis_frame.CSHD;
            individual_data.FSHD = Marker_in_pelvis_frame.FSHD;
            individual_data.LSHD = Marker_in_pelvis_frame.LSHD;
            individual_data.RSHD = Marker_in_pelvis_frame.RSHD;

        case 'no_head'
            individual_data.LANK = Marker_in_pelvis_frame.LANK;
            individual_data.RANK = Marker_in_pelvis_frame.RANK;
            individual_data.RKNE = Marker_in_pelvis_frame.RKNE;
            individual_data.LKNE = Marker_in_pelvis_frame.LKNE;
            individual_data.LPEL = Marker_in_pelvis_frame.LPEL;
            individual_data.RPEL = Marker_in_pelvis_frame.RPEL;
            individual_data.LSHO = Marker_in_pelvis_frame.LSHO;
            individual_data.RSHO = Marker_in_pelvis_frame.RSHO;
        
            individual_data.RELB = Marker_in_pelvis_frame.RELB;
            individual_data.LELB = Marker_in_pelvis_frame.LELB;
            individual_data.RWRA = Marker_in_pelvis_frame.RWRA;
            individual_data.LWRA = Marker_in_pelvis_frame.LWRA;
    end
    
    if laterality == "right"
        individual_data.ankle_pos = Marker_in_pelvis_frame.RANK;

    else
        individual_data.ankle_pos = Marker_in_pelvis_frame.LANK;
    end

    results.(bambiID) = individual_data;

    outcome_result = fullfile(outcome_path, 'resultat_simple.mat');
    save(outcome_result, 'results');
    
    clear markerData Marker_in_pelvis_frame Marker_in_pelvis_frame_plot M_f GapfilledDataSet individual_data
end
