%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BAMBI Supine-Based Kinematic Processing Script
%
% Description:
% This script processes 3D kinematic data from C3D motion capture files 
% collected for the BAMBI project, focusing on grouping trials by supine 
% positions (e.g., Supine1, Supine2, etc.) for each participant. It includes 
% marker set categorization, gap filling, low-pass filtering, and 
% concatenation of trials within each supine group.
%
% Workflow Overview:
% 1. Scan the base directory to list all C3D files.
% 2. Extract unique participant IDs from filenames.
% 3. Load participant metadata (e.g., laterality) from a CSV file.
% 4. For each participant:
%    - Group all C3D files by supine session label (e.g., Supine1).
%    - For each file in a supine group:
%        - Read marker data using BTK.
%        - Identify the marker set configuration ('full', 'no_knee', etc.).
%        - Replace missing values (zeros) with NaNs.
%        - Apply gap-filling to estimate missing marker data.
%        - Filter marker trajectories using a zero-phase low-pass 
%          Butterworth filter (6 Hz cutoff).
%    - Concatenate data across all trials in the supine group.
%    - Store marker data and ankle position (based on laterality) 
%      for each participant and supine condition.
% 5. Save structured results grouped by participant and supine group 
%    in a .mat file.
%
% Inputs:
% - C3D motion capture files stored in a specified directory.
% - A CSV file containing participant metadata (inclusion numbers, laterality).
%
% Outputs:
% - A .mat file ('resultat_grouped_by_supine.mat') containing processed
%   marker data for each participant, organized by supine session.
%
% Dependencies:
% - BTK toolbox for reading C3D files.
% - Custom functions: check_category, PredictMissingMarkers
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc
addpath('function')
plot_fig=1;
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');

%% === Define paths ===
outcome_path = 'S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\OutcomeRaw';

base_path = 'S:\KLab\#SHARE\RESEARCH\BAMBI\Data\Kinematic analysis\3 months_validity and reliability';

% Liste tous les fichiers dans le dossier
files = dir(fullfile(base_path, '*')); 

% Initialise une cellule pour stocker les noms extraits
extractedNames = {};

results = struct();

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
        expected_markers.no_upperlimb = {'LANK','LKNE','LPEL','LSHO','RANK','RKNE','RPEL','RSHO'};
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

        %% === Time Duration ===
        n = size(M_f.RKNE, 1);     % Number of time frames
        time = (0:n-1) / frameRate;                  % Time vector in seconds
        individual_data.time_duration = time;        % Store time duration



    % Define virtual markers (midpoints)
        midShoulder  = (M_f.LSHO + M_f.RSHO) / 2;
        midLPelSho   = (M_f.LSHO + M_f.LPEL) / 2;
        midRPelSho   = (M_f.RSHO + M_f.RPEL) / 2;
        midPelvis    = (M_f.LPEL + M_f.RPEL) / 2;
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
        list_marker = fieldnames(M_f);   
        n_marker = length(list_marker);                        
        
        for m = 1:n_marker
            % Prepare marker for transformation (homogeneous coordinates)
            tmp1 = [permute(M_f.(list_marker{m}), [2 3 1]); ones(1,1,n)];
            tmp2 = Mprod_array3(Tinv_array3(T_gt), tmp1);           % Apply inverse transform
            m_th.(list_marker{m}) = permute(tmp2(1:3,1,:), [3 1 2]);        % Save transformed marker
            Marker_in_pelvis_frame.(list_marker{m}) = m_th.(list_marker{m});        % For plotting/export
        end
        
        % Prepare markers for 2D plotting (XY plane)
        marker_list_to_plot = expected_markers.full;
        Marker_in_pelvis_frame_plot.X = [];
        Marker_in_pelvis_frame_plot.Y = [];
        
        for m = 1:length(marker_list_to_plot)
            marker_name = marker_list_to_plot{m};
            if isfield(Marker_in_pelvis_frame, marker_name)
                Marker_in_pelvis_frame_plot.X(:, end+1) = Marker_in_pelvis_frame.(marker_name)(:,1);
                Marker_in_pelvis_frame_plot.Y(:, end+1) = Marker_in_pelvis_frame.(marker_name)(:,2);
            end
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
        saveas(gcf, fullfile(outcome_folder, [erase(c3d_filenames(j).name, '.c3d') '_2d_plot.png']));
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

                individual_data.RANK_global_frame = M_f.RANK;
                individual_data.LANK_global_frame = M_f.LANK;

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

                individual_data.RANK_global_frame = M_f.RANK;
                individual_data.LANK_global_frame = M_f.LANK;
        end
        
        if laterality == "right"
            individual_data.ankle_pos = Marker_in_pelvis_frame.RANK;
    
        else
            individual_data.ankle_pos = Marker_in_pelvis_frame.LANK;
        end
    
        results.(erase(c3d_filenames(j).name, '.c3d')) = individual_data;
    
        fprintf('File %s finished \n', c3d_filenames(j).name);
    
    
    clear acq M markers midShoulder midLPelSho midRPelSho midPelvis center ...
      xtmp y_gt z_gt x_gt O_gt R_gt T_gt tmp1 tmp2 m_th list_marker ...
      n_marker category missing_markers frameRate butter_B butter_A ...
      csv_tab indices laterality Marker_in_pelvis_frame_plot M_f Marker_in_pelvis_frame
    end
end

outcome_result = fullfile(outcome_path, 'resultat_no_combined.mat');
save(outcome_result, 'results');
