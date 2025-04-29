## Matlab Part

This repository contains MATLAB scripts for analyzing 3D motion capture data (.c3d) using the **BTK Toolbox**. The analysis is designed for "Bambi" participants and follows a two-step process:

1. **Step 1 – Max Activity Window Detection**  
   Identify the 5-second time window with **maximum movement intensity** (based on ankle speed).

2. **Step 2 – Kinematic Outcome Extraction**  
   Extract and analyze biomechanical outcomes **within the selected window**.

---

## Requirements

- MATLAB (R2016b or newer)
- [BTK Toolbox for MATLAB](https://github.com/Biomechanical-ToolKit/BTKCore)

### Installation

Install BTK and add it to your MATLAB path:
```matlab
addpath(genpath('path_to_btk_folder'));
savepath;
```

---

## Step 1 – Max Activity Window Detection

**Script**: `Bambi_treat_all_time.m`

Determine the window with the **highest movement intensity** for each participant, based on the Area Under the Curve (AUC) of foot speed.

### Outputs

- One plot per participant showing:
  - Complete speed trace
  - Highlighted max-activity window
- An Excel file: `max_activity_windows.xlsx` with start and end frame of the selected window

---

## Step 2 – Kinematic Outcome Extraction

**Script**: `Bambi_Kinematics_outcomes_total.m`

Using the window computed in Step 1, extract joint kinematics and movement features for each participant.

### Outputs

- A Mat file: `resultats.mat` with hip joint kinematics and trajectory marker for each bambi

---

## Execution Order

> ⚠️ You must run the scripts in the following order:

1. `Bambi_treat_all_time.m` → generates `AUC_windows_results.xlsx`
2. `Bambi_Kinematics_outcomes_total.m` → uses the computed window to extract outcomes and generates `resultats.mat`

---

## Python Part

### Requirements

To run the Python script, you'll need the following:

- **Python 3.x**

### Required Python packages:
- `scipy`
- `matplotlib`
- `pandas`
- `numpy`
- `sklearn`
- `seaborn`
- `scipy.stats` 
- `scipy.interpolate` 
- `Function.Base_function`

### Installation Instructions

You can install the required dependencies using pip:

```bash
pip install scipy matplotlib pandas numpy scikit-learn seaborn
```

### Custom Module

Ensure that the custom function module `Base_function` is available and properly imported in the project directory. This module contains essential functions for plotting and other statistical calculations.

### Outputs

- Mean PDF plots (across all participants) for hip adduction/abduction and flexion/extension
- Individual ankle ellipsoid plots (3D spatial distribution) per participant
- A CSV file summarizing per-participant kinematic and movement statistics, including:
  - Hip angles: mean, std, skewness, kurtosis, mode (for adduction and flexion)
  - Ankle: total distance traveled, velocity skewness
  - Ellipsoid: number of points, number enclosed, % enclosed, and 90% volume
