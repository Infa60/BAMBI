import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew
from Function.Ellipsoid import (
    plot_ellipsoid_and_points_stickman
)
from Function.Plot import (
    plot_combined_pdf,
    plot_mean_pdf_stat
)
from Function.Kicking_function import kicking, get_mean_and_std
from Function.Members_contact import distance_foot_foot, distance_hand_hand

# Set matplotlib backend
matplotlib.use("TkAgg")

# Set path and load .mat file
path = "/Users/mathieubourgeois/Documents/BAMBI_Data"
result_file = f"{path}/resultats.mat"
data = scipy.io.loadmat(result_file)

# Access the structured results
results_struct = data["results"][0, 0]

# Lists to collect data for global analysis
data_rows = []
hip_add_all = []
hip_flex_all = []
pdf_list = []
x_list = []
bambiID_list = results_struct.dtype.names  # Extract all Bambi IDs

# Iterate through each Bambi ID
for i, bambiID in enumerate(results_struct.dtype.names):
    row = {}
    row["bambiID"] = bambiID

    # Extract statistical features for hip adduction/abduction
    row['mean_hip_angle_add'] = results_struct[bambiID]['mean_hip_angle_add'][0, 0].item()
    row['std_hip_angle_add'] = results_struct[bambiID]['std_hip_angle_add'][0, 0].item()
    row['skew_hip_angle_add'] = results_struct[bambiID]['skew_hip_angle_add'][0, 0].item()
    row['kurt_hip_angle_add'] = results_struct[bambiID]['kurt_hip_angle_add'][0, 0].item()
    row['mode_add'] = results_struct[bambiID]['mode_add'][0, 0].item()

    # Extract statistical features for hip flexion/extension
    row['mean_hip_angle_flex'] = results_struct[bambiID]['mean_hip_angle_flex'][0, 0].item()
    row['std_hip_angle_flex'] = results_struct[bambiID]['std_hip_angle_flex'][0, 0].item()
    row['skew_hip_angle_flex'] = results_struct[bambiID]['skew_hip_angle_flex'][0, 0].item()
    row['kurt_hip_angle_flex'] = results_struct[bambiID]['kurt_hip_angle_flex'][0, 0].item()
    row['mode_flex'] = results_struct[bambiID]['mode_flex'][0, 0].item()

    # Append angle data to list for further PDF plotting
    hip_add = results_struct[bambiID]["hip_angle_add"][0, 0]
    hip_add_all.append(hip_add)
    hip_flex = results_struct[bambiID]["hip_angle_flex"][0, 0]
    hip_flex_all.append(hip_flex)

    # Retrieve joint positions
    pos_ankle = results_struct[bambiID]["ankle_pos"][0, 0]
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
    time_duration = results_struct[bambiID]["time_duration"][0][0][0]

    # Calculate total distance traveled by the ankle
    distances = np.linalg.norm(np.diff(pos_ankle, axis=0), axis=1)
    row["distance_travel_ankle"] = np.sum(distances)

    # Call function to plot ellipsoid and stickman, retrieve geometric stats
    stats = plot_ellipsoid_and_points_stickman(
        pos_ankle,
        RANK,
        LANK,
        RKNE,
        LKNE,
        RPEL,
        LPEL,
        RSHO,
        LSHO,
        RELB,
        LELB,
        LWRA,
        RWRA,
        bambiID,
        folder_save_path=f"{path}/Outcomes_plot",
        interactive=False,
        inside_point=False,
        outside_point=False,
    )

    # Save geometric and velocity distribution stats
    row["num_points"] = stats["num_points"]
    row["num_enclosed"] = stats["num_enclosed"]
    row["percentage_enclosed"] = stats["percentage_enclosed"]
    row["volume_90"] = stats["volume_90"]
    row["skew_velocity"] = skew(results_struct[bambiID]["velocity_ankle"][0, 0]).item()

    # Add the row to the list
    data_rows.append(row)

    ## Add in outcome
    kicking_cycle_data_left, distance_kicking_left = kicking(LPEL,LANK, time_duration, 1)
    mean_std_kicking_values_left = get_mean_and_std(kicking_cycle_data_left)

    kicking_cycle_data_right, distance_kicking_right = kicking(RPEL,RANK, time_duration, 1)
    mean_std_kicking_values_right = get_mean_and_std(kicking_cycle_data_right)

    n_events, total_time, event_durations = distance_foot_foot(LANK, RANK, LKNE, RKNE, threshold=100, time_vector=time_duration)

    n_events, total_time, event_durations = distance_hand_hand(LWRA, RWRA, threshold=100, time_vector=time_duration, plot=True)

    print(n_events, event_durations)


# Convert all collected data into a DataFrame
df = pd.DataFrame(data_rows)

# Save DataFrame to CSV
df.to_csv(f"{path}/Final_results.csv", index=False)
print("Excel file successfully created!")

# Plot PDFs and mean statistics for hip adduction-abduction
plot_combined_pdf(hip_add_all, "Adduction-Abduction", folder_save_path=path)
plot_mean_pdf_stat(
    hip_add_all,
    bambiID_list,
    "Adduction-Abduction",
    folder_save_path=path,
    grid_min=-70,
    grid_max=45,
    grid_points=500,
)

# Plot PDFs and mean statistics for hip flexion-extension
plot_combined_pdf(hip_flex_all, "Flexion-Extension", folder_save_path=path)
plot_mean_pdf_stat(
    hip_flex_all,
    bambiID_list,
    "Flexion-Extension",
    folder_save_path=path,
    grid_min=-10,
    grid_max=80,
    grid_points=500,
)
