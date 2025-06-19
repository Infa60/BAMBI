import scipy.io
from scipy.stats import skew
from collections import Counter
from PythonFunction.Ellipsoid import (
    plot_ellipsoid_and_points_stickman
)
from PythonFunction.Leg_lift_adduct import (
    plot_combined_pdf,
    plot_mean_pdf_stat,
    ankle_high
)
from PythonFunction.Kicking_function import *
from PythonFunction.Members_contact import *
from PythonFunction.Body_symmetry import *
from PythonFunction.Head_contact_orientation import *
from PythonFunction.Base_function import *
from PythonFunction.TEST import *

# Set matplotlib backend
matplotlib.use("TkAgg")

# Set path and load .mat file
path = "/Users/mathieubourgeois/Documents/BAMBI_Data"
result_file = f"{path}/resultats.mat"
anthropo_file = f"{path}/REDcap_template_3months.csv"

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
    if results_struct[bambiID]['marker_category'][0][0][0] != "full":
        continue

    print(f"{bambiID} is running")

    leg_length, shank_length = get_leg_and_tibia_length(anthropo_file, bambiID)

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
    CSHD = results_struct[bambiID]["CSHD"][0, 0]
    FSHD = results_struct[bambiID]["FSHD"][0, 0]
    LSHD = results_struct[bambiID]["LSHD"][0, 0]
    RSHD = results_struct[bambiID]["RSHD"][0, 0]

    time_duration = results_struct[bambiID]["time_duration"][0][0][0]
    while isinstance(time_duration, np.ndarray) and time_duration.ndim > 1:
        time_duration = time_duration[0]
    cycle_durations = np.diff(time_duration)
    freq = int(1 / cycle_durations[0])

    body_length = get_body_length(RPEL, LPEL, RSHO, LSHO)

    # Calculate total distance traveled by the ankle
    distances = np.linalg.norm(np.diff(pos_ankle, axis=0), axis=1)
    row["distance_travel_ankle"] = np.sum(distances)

    # Call function to plot ellipsoid and stickman, retrieve geometric stats
    stats_ankle_ellipsoid = plot_ellipsoid_and_points_stickman(
        pos_ankle, RANK, LANK, RKNE, LKNE, RPEL, LPEL, RSHO, LSHO, RELB, LELB, LWRA, RWRA, CSHD, FSHD, LSHD, RSHD, bambiID,
        folder_save_path=f"{path}/Outcomes_plot",
        confidence_threshold=0.90,
        interactive=False,
        inside_point=False,
        outside_point=False,
    )

    # Save geometric and velocity distribution stats
    row["num_points"] = stats_ankle_ellipsoid["num_points"]
    row["num_enclosed"] = stats_ankle_ellipsoid["num_enclosed"]
    row["percentage_enclosed"] = stats_ankle_ellipsoid["percentage_enclosed"]
    row["volume_90"] = stats_ankle_ellipsoid["volume_90"]
    row["skew_velocity"] = skew(results_struct[bambiID]["velocity_ankle"][0, 0]).item()

    # Add the row to the list
    data_rows.append(row)

    ## Hand to mouth contact
    distance_hand_mouth(LWRA, RWRA, CSHD, FSHD, LSHD, RSHD, threshold=100, time_vector=time_duration, plot=False)

    ## Hand-hand contact
    hand_hand_contact = distance_hand_hand(LWRA, RWRA, threshold=50, time_vector=time_duration, plot=False)

    ## Kicking
    kicking_cycle_outcomes_left, distance_kicking_left, kick_intervals_left = kicking(LPEL,LANK, time_duration, leg_length, LKNE, LPEL, LANK, freq, plot=False)
    mean_std_kicking_values_left = get_mean_and_std(kicking_cycle_outcomes_left)

    kicking_cycle_outcomes_right, distance_kicking_right, kick_intervals_right = kicking(RPEL,RANK, time_duration, leg_length, RKNE, RPEL, RANK, freq, plot=False)
    mean_std_kicking_values_right = get_mean_and_std(kicking_cycle_outcomes_right)

    knee_angle_right, hip_angle_right = synchro_hip_knee(time_duration, RPEL, RKNE, RSHO, RANK, plot=False)
    knee_angle_left, hip_angle_left = synchro_hip_knee(time_duration, LPEL, LKNE, LSHO, LANK, plot=False)

    # knee_hip_correlation(knee_angle_right, hip_angle_right, kick_intervals_right)

    #phase_antiphase(knee_angle_right, hip_angle_right, time_duration)
    #phase_antiphase(knee_angle_left, hip_angle_left, time_duration)

    #phase_antiphase(knee_angle_right, knee_angle_left, time_duration)

    classification_results = classify_kicks(kick_intervals_right, kick_intervals_left, knee_angle_right, knee_angle_left, fs=freq)

    type_counts = Counter(r['type'] for r in classification_results)
    for k, v in type_counts.items():
        print(f"{k}: {v}")

    plot_kick_classification_with_bars(knee_angle_right, knee_angle_left, classification_results, freq)

    # Liste de sauvegarde
    labeled_kicks = []

    # Exemple : parcourir tous les kicks droits
    for (start, end) in kick_intervals_right:
        label_and_save_kick(knee_angle_right, knee_angle_left, start, end, 'right', labeled_kicks)

    plot_kick_classification_with_bars(knee_angle_right, knee_angle_left, labeled_kicks, freq)


    ## Foot-foot contact
    plantar_plantar_contact_outcomes, foot_foot_contact_outcomes = distance_foot_foot(LANK, RANK, LKNE, RKNE, threshold_ankle=150, threshold_knee=300, time_vector=time_duration, plot=False)

    ## Leg lifting
    right_lift_with_leg_extend, distance_pelv_ank_right = ankle_high(RANK, RPEL, time_vector=time_duration, leg_length=leg_length, high_threshold=80, max_flexion=30, plot=False)
    left_lift_with_leg_extend, distance_pelv_ank_left = ankle_high(LANK, LPEL, time_vector=time_duration, leg_length=leg_length, high_threshold=80, max_flexion=30, plot=False)

    # phase_antiphase(distance_pelv_ank_right, distance_pelv_ank_left, time_duration)

    ## Hand-foot contact
    hand_foot_contact_outcomes = distance_hand_foot(LANK, RANK, LWRA, RWRA, threshold=100, time_vector=time_duration, plot=False)

    ## Wrist ellipsoid
    stats_right_wrist_ellipsoid = plot_ellipsoid_and_points_stickman(
        RWRA, RANK, LANK, RKNE, LKNE, RPEL, LPEL, RSHO, LSHO, RELB, LELB, LWRA, RWRA, CSHD, FSHD, LSHD, RSHD, bambiID,
        folder_save_path=f"{path}/Right_Wrist_Outcomes_plot",
        confidence_threshold=0.99,
        interactive=False,
        inside_point=False,
        outside_point=False,
    )
    stats_left_wrist_ellipsoid = plot_ellipsoid_and_points_stickman(
        LWRA, RANK, LANK, RKNE, LKNE, RPEL, LPEL, RSHO, LSHO, RELB, LELB, LWRA, RWRA, CSHD, FSHD, LSHD, RSHD, bambiID,
        folder_save_path=f"{path}/Left_Wrist_Outcomes_plot",
        confidence_threshold=0.99,
        interactive=False,
        inside_point=False,
        outside_point=False,
    )

    ## Head rotation
    head_rotation(CSHD, FSHD, LSHD, RSHD, LSHO, RSHO, LPEL, RPEL, threshold=(-5,5), time_vector=time_duration, plot=False)

    ## Body symmetry
    body_symmetry(LPEL, RPEL, LSHO, RSHO,40, time_vector=time_duration, plot=False)


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
