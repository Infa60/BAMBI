"""
Manual Kick Phase Labeling - Interactive Script
===============================================

This script allows you to visualize, review, and manually annotate detected knee extension peaks
in time-series data (e.g., for gait or rehabilitation studies). It processes a structured .mat
results file (one entry per BAMBI ID), applies a Butterworth filter, detects peaks, and opens
an interactive plot for labeling each peak as "start", "end", or "end+start" (for phase transitions).
Your labels are exported as a CSV for each subject/session.

Main Features:
--------------
- Loads joint data and timing information from a .mat structure.
- Computes filtered knee angle and detects peaks automatically.
- Skips any session already annotated (CSV exists) if "continue_to_last" is set to True.
- Interactive annotation: color-coded, validated, and user-friendly.
- Exports a CSV file per BAMBI ID, containing "start" and "end" frame indices.

How to Use:
-----------
1. Ensure you have the required packages: numpy, scipy, matplotlib, pandas.
2. Update `path` and utility function imports if needed.
3. Run the script. For each BAMBI ID, an interactive window opens for labeling (unless already labeled and skipping is enabled).

Keyboard Shortcuts (in the interactive window):
    ←/→ : Move to previous/next peak
    a   : Label as "start"         (green)
    z   : Label as "end"           (red)
    e   : Label as "end+start"     (orange)
    d   : Hide this peak           (light grey, not exported)
    m   : Save CSV (no dialog, automatically named in /Kick_detection/)

Rules:
------
- Alternation enforced: first label must be "start", then alternate (start → end/end+start → ...).
- Never assign two consecutive "start" labels.
- Peaks without any label (still black) are not exported.
- Hidden peaks ("d") are skipped and not exported. (Note: Once a peak is hidden, it cannot be changed except by reloading the script.)
- Files already labeled are skipped on rerun (you can disable this by setting "continue_to_last" to False).
- On saving, a CSV with columns "start" and "end" (frame indices) is exported.

Tips:
-----
- You can re-run the script to reload or revise labels (especially if you accidentally hide a peak).
"""

import scipy.io
import matplotlib
import os
from PythonFunction.Kick_detection_manually import *
from PythonFunction.Kicking_function import *

# Set matplotlib backend for interactive use
matplotlib.use("TkAgg")

# Set paths and load the .mat file
path = "/Users/mathieubourgeois/Documents/BAMBI_Data"
outcome_folder = f"{path}/Kick_detection"
os.makedirs(outcome_folder, exist_ok=True)

continue_to_last = True

result_file = f"{path}/resultat_no_combined_pelvis_frame.mat"
data = scipy.io.loadmat(result_file)

# Access the structured results
results_struct = data["results"][0, 0]
bambiID_list = results_struct.dtype.names  # All Bambi IDs

# Iterate through each Bambi ID
for i, bambiID in enumerate(results_struct.dtype.names):

    print(f"{bambiID} is running")
    print(results_struct[bambiID]['marker_category'][0][0][0])

    bambi_name = bambiID.split("_", 1)[0]

    # Extract time and frequency information
    time_duration = results_struct[bambiID]["time_duration"][0][0][0]
    while isinstance(time_duration, np.ndarray) and time_duration.ndim > 1:
        time_duration = time_duration[0]
    cycle_durations = np.diff(time_duration)
    freq = int(1 / cycle_durations[0])

    total_time_sec = time_duration[-1]
    total_time_min = total_time_sec / 60.0

    out_path = f"{bambiID}_cleaned.csv"
    outcome_path = os.path.join(outcome_folder, out_path)

    # Skip if already labeled and continue_to_last is True
    if continue_to_last:
        if os.path.exists(outcome_path):
            print(f"Already exists, skipping: {outcome_path}")
            continue

    # Retrieve joint positions
    RANK = results_struct[bambiID]["RANK"][0, 0]
    LANK = results_struct[bambiID]["LANK"][0, 0]
    RKNE = results_struct[bambiID]["RKNE"][0, 0]
    LKNE = results_struct[bambiID]["LKNE"][0, 0]
    LPEL = results_struct[bambiID]["LPEL"][0, 0]
    RPEL = results_struct[bambiID]["RPEL"][0, 0]
    LSHO = results_struct[bambiID]["LSHO"][0, 0]
    RSHO = results_struct[bambiID]["RSHO"][0, 0]
    LELB = results_struct[bambiID]["LELB"][0, 0]
    RELB = results_struct[bambiID]["RELB"][0, 0]
    LWRA = results_struct[bambiID]["LWRA"][0, 0]
    RWRA = results_struct[bambiID]["RWRA"][0, 0]
    CSHD = results_struct[bambiID]["CSHD"][0, 0]
    FSHD = results_struct[bambiID]["FSHD"][0, 0]
    LSHD = results_struct[bambiID]["LSHD"][0, 0]
    RSHD = results_struct[bambiID]["RSHD"][0, 0]

    # Compute the knee angle and apply Butterworth filter
    knee_angle = 180 - angle_from_vector(RKNE, RPEL, RANK)
    cutoff = 6  # Hz
    knee_angle_filt = butter_lowpass_filter(knee_angle, cutoff, freq, order=2)

    # Detect extension peaks
    peaks, _ = find_peaks(
        knee_angle_filt,
        prominence=0.01 * (np.max(knee_angle_filt) - np.min(knee_angle_filt))
    )

    # Interactive labeling
    interactive_label_peaks(
        knee_angle_filt,
        peaks,
        out_path=outcome_path,  # CSV will be saved automatically here
        fs=freq
    )
