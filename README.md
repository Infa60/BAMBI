-This repository contains MATLAB and Python scripts for analyzing 3D motion capture data (`.c3d`) using the **BTK Toolbox** and custom analysis workflows. The project is designed to process trials for participants (referred to as "Bambi") and extract meaningful kinematic and movement-related outcomes.
-
----
-
-## MATLAB Part
-
-The MATLAB analysis is divided into **two main steps**:
-
-1. **Step 1 – Max Activity Window Detection**  
-   Identifies the time window (rounded to the nearest 5 seconds) with the **maximum movement intensity**, based on ankle speed.
-
-2. **Step 2 – Kinematic Outcome Extraction**  
-   Computes joint angles and other biomechanical outcomes **within the selected window**.
-
-
-## Requirements
-
-- MATLAB (R2016b or newer)
-- [BTK Toolbox for MATLAB](https://github.com/Biomechanical-ToolKit/BTKCore)
-
-### Installation
-
-Install BTK and add it to your MATLAB path:
-```matlab
-addpath(genpath('path_to_btk_folder'));
-savepath;
-```
-
-## Step 1 – Max Activity Window Detection
-
-**Script**: `Bambi_treat_all_time.m`
-
-This script determines the window of highest movement intensity for each participant using the Area Under the Curve (AUC) of foot speed.
-
-### Outputs
-
-- One plot per participant showing:
-   - Full ankle speed trace
-   - Highlighted max-activity window
-- AUC_windows_results.xlsx: Start and end frames for the selected window
-
-## Step 2 – Kinematic Outcome Extraction
-
-**Script**: `Bambi_Kinematics_outcomes_total.m`
-
-This script uses the selected max-activity window to extract joint kinematics and marker data.
-
-
-### Outputs
-
-- `resultats.mat`: A .mat file with joint angles, marker trajectories, and velocity profiles for each participant
-
-
-## Execution Order
-
-You must run the scripts in the following order:
-
-1. `Bambi_treat_all_time.m` → generates `AUC_windows_results.xlsx`
-2. `Bambi_Kinematics_outcomes_total.m` → uses the computed window to extract outcomes and generates `resultats.mat`
-
----
-
-## Python Part
-
-### Requirements
-
-To run the Python script, you'll need the following:
-
-- **Python 3.x**
-
-### Required Python Packages
-
-You can install the required dependencies using pip:
-
+# BAMBI
+
+## Overview
+This project provides a processing pipeline for 3‑D motion capture trials stored as `.c3d` files. It aims to extract kinematic indicators from recordings performed by subjects nicknamed "Bambi".
+
+## Repository structure
+- `Function/` – MATLAB functions used for marker processing and matrix calculations.
+- `PythonFunction/` – Python modules for statistics and visualization.
+- Key MATLAB scripts:
+  - `Bambi_treat_all_time.m`: detect the time window with maximal activity.
+  - `Bambi_gap_extract.m` and `Bambi_combine_per_bambi_gap_extract.m`: compute indicators inside the detected window.
+- Key Python scripts:
+  - `bambi_results_processing.py`: post-process results and generate figures.
+  - `bambi_detect_kick_manual.py`: manually flag events (e.g., kicks).
+
+## Prerequisites
+### MATLAB
+- MATLAB R2016b or later
+- [BTK toolbox](https://github.com/Biomechanical-ToolKit/BTKCore)
+
+### Python
+- Python 3.x
+- Libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
+
+Install Python dependencies:
 ```bash
-pip install scipy matplotlib pandas numpy scikit-learn seaborn
+pip install numpy pandas scipy scikit-learn matplotlib seaborn
 ```
-Make sure the custom module Function.Base_function is available in your Python environment. It contains utility functions for plotting and statistical analysis.
-
-
-### Outputs
-
-- Mean PDF plots (across all participants) for hip adduction/abduction and flexion/extension
-- Individual ankle ellipsoid plots (3D spatial distribution) per participant
-- A CSV file summarizing per-participant kinematic and movement statistics, including:
-  - Hip angles: mean, std, skewness, kurtosis, mode (for adduction and flexion)
-  - Ankle: total distance traveled, velocity skewness
-  - Ellipsoid: number of points, number enclosed, % enclosed, and 90% volume
 
+## Recommended workflow
+1. **Detect the maximal activity window** with `Bambi_treat_all_time.m`.
+2. **Extract kinematic indicators** within this window using `Bambi_gap_extract.m` or `Bambi_combine_per_bambi_gap_extract.m`.
+3. **Post-process in Python** using `bambi_results_processing.py` and modules in `PythonFunction/` to run statistical analyses and create visualizations.
 
+## Expected outcomes
+Running the workflow yields:
+- Segmented trials with the time interval of maximal movement.
+- Tables or `.mat`/`.csv` files containing extracted kinematic indicators per trial.
+- Statistical summaries and plots (boxplots, correlation graphs, symmetry metrics, etc.) produced by Python scripts.
+These outcomes help evaluate motion patterns and compare participants.
 
-- Mean PDF plots across all participants for:
-   - Hip adduction/abduction
-   - Hip flexion/extension
-- 3D ankle ellipsoid plots for each participant
-- CSV summary of movement and kinematic metrics per participant:
-   - Hip joint angles (add/flex): mean, std, skewness, kurtosis, mode
-   - Ankle: distance traveled, velocity skewness
-   - Ellipsoid: number of points, number enclosed, percentage enclosed, 90% volume
+## Contributing
+Contributions are welcome! Please open an issue to discuss ideas or submit a pull request with a clear description of your changes.
 
+## License
+This project is distributed under the MIT License.
